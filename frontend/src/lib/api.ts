import axios from 'axios';

// Create axios instance for the backend API
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface UploadImageParams {
  file: File;
  description?: string;
  customMetadata?: string;
  removeBg?: boolean;
}

export interface SearchImageParams {
  file: File;
  filters?: string[];
  limit?: number;
}

export interface SearchTextParams {
  query: string;
  filters?: string[];
  limit?: number;
}

export interface SearchMultimodalParams {
  file: File;
  query: string;
  weightImage?: number;
  filters?: string[];
  limit?: number;
}

export interface UpdateMetadataParams {
  imageId: string;
  description: string;
  customMetadata?: string;
}

// API client for the image similarity search backend
const apiClient = {
  // Image Upload
  async uploadImage({ file, description, customMetadata, removeBg = false }: UploadImageParams) {
    const formData = new FormData();
    formData.append('file', file);
    
    if (description) {
      formData.append('description', description);
    }
    
    if (customMetadata) {
      formData.append('custom_metadata', customMetadata);
    }
    
    formData.append('remove_bg', String(removeBg));
    
    return api.post('/api/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },
  
  // Folder Upload with Progress Monitoring
  async uploadFolderWithProgress({ 
    files, 
    removeBg = false,
    onProgress = () => {}
  }: { 
    files: File[], 
    removeBg: boolean,
    onProgress?: (status: string, currentIndex: number, totalFiles: number, filterInfo?: {
      currentFilter?: string;
      filterIndex?: number;
      totalFilters?: number;
    }) => void
  }) {
    // Total files to process
    const totalFiles = files.length;
    
    // Upload files one by one to track progress
    const results = {
      success: true,
      total: totalFiles,
      successful: 0,
      skipped: 0,
      failed: 0,
      results: [] as Array<{
        filename: string;
        status: string;
        id?: string;
        reason?: string;
      }>
    };
    
    // Declare filterPollingInterval at this scope to ensure we can clear it
    let filterPollingInterval: NodeJS.Timeout | null = null;
    
    // Function to clean up the polling interval
    const cleanupPolling = () => {
      if (filterPollingInterval) {
        clearInterval(filterPollingInterval);
        filterPollingInterval = null;
      }
    };
    
    try {
      // Process each file
      for (let i = 0; i < totalFiles; i++) {
        const file = files[i];
        onProgress(`Uploading file ${i + 1} of ${totalFiles}: ${file.name}`, i, totalFiles);
        
        try {
          // Upload single file
          const formData = new FormData();
          formData.append('file', file);
          formData.append('remove_bg', String(removeBg));
          
          // Start a mock polling of filter application for this image
          // This isn't real filter progress, but it gives the user feedback while waiting
          // First, clean up any existing interval
          cleanupPolling();
          
          let mockFilterIndex = 0;
          let mockTotalFilters = 0;
          
          // Start polling to see if filters are being applied
          filterPollingInterval = setInterval(() => {
            // Check if we have an ID and filters to apply
            api.get('/api/filters')
              .then(response => {
                const filters = response.data.filters || [];
                mockTotalFilters = filters.length;
                
                if (mockTotalFilters > 0) {
                  if (mockFilterIndex < mockTotalFilters) {
                    mockFilterIndex++;
                  }
                  
                  onProgress(
                    `Processing file ${i + 1}/${totalFiles}: Applying filters (${mockFilterIndex}/${mockTotalFilters})`,
                    i,
                    totalFiles,
                    {
                      currentFilter: filters[mockFilterIndex - 1] || "Unknown filter",
                      filterIndex: mockFilterIndex,
                      totalFilters: mockTotalFilters
                    }
                  );
                }
              })
              .catch(err => {
                console.error("Error checking filters:", err);
              });
          }, 1000);
          
          const response = await api.post('/api/upload', formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
          });
          
          // Clear the filter polling interval
          cleanupPolling();
          
          // Add result
          results.successful++;
          results.results.push({
            filename: file.name,
            status: 'success',
            id: response.data.metadata?.id
          });
          
          // Update progress with success
          onProgress(`Processed ${i + 1} of ${totalFiles}: ${file.name} (Success)`, i, totalFiles);
          
        } catch (error: unknown) {
          // Clean up polling interval if there's an error
          cleanupPolling();
          
          // Check if it's a duplicate
          const err = error as { 
            response?: { 
              status?: number; 
              data?: { 
                error?: string; 
                metadata?: {
                  id: string;
                  filename: string;
                  description: string;
                  url: string;
                  thumbnail_url?: string;
                  processed_url?: string;
                  custom_metadata?: string;
                  created_at?: string;
                  [key: string]: string | number | boolean | undefined;
                };
                message?: string;
              } 
            } 
          };
          
          if (err.response?.status === 409 || 
              err.response?.data?.error?.includes('Duplicate')) {
            results.skipped++;
            results.results.push({
              filename: file.name,
              status: 'skipped',
              reason: err.response?.data?.message || 'Duplicate image',
              id: err.response?.data?.metadata?.id // Extract the ID of the duplicate
            });
            onProgress(`Processed ${i + 1} of ${totalFiles}: ${file.name} (Skipped - Duplicate)`, i, totalFiles);
          } else {
            results.failed++;
            results.results.push({
              filename: file.name,
              status: 'error',
              reason: err.response?.data?.error || 'Unknown error'
            });
            onProgress(`Processed ${i + 1} of ${totalFiles}: ${file.name} (Failed)`, i, totalFiles);
          }
        }
        
        // Small delay to avoid overwhelming the server
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      
      onProgress('Complete', totalFiles, totalFiles);
    } finally {
      // Ensure polling is cleaned up no matter what
      cleanupPolling();
    }
    
    return { data: results };
  },
  
  // Standard Folder Upload (uses the batch endpoint)
  async uploadFolder({ files, removeBg = false }: { files: File[], removeBg: boolean }) {
    const formData = new FormData();
    
    files.forEach(file => {
      formData.append('files', file);
    });
    
    formData.append('remove_bg', String(removeBg));
    
    return api.post('/api/upload-folder', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },
  
  // Image Search
  async searchByImage({ file, filters, limit }: SearchImageParams) {
    const formData = new FormData();
    formData.append('file', file);
    
    if (filters && filters.length > 0) {
      filters.forEach(filter => {
        formData.append('filters', filter);
      });
    }
    
    // Always append limit, including when it's 0 (All)
    formData.append('limit', String(limit !== undefined ? limit : 10));
    
    return api.post('/api/search/image', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },
  
  // Text Search
  async searchByText({ query, filters, limit }: SearchTextParams) {
    const formData = new FormData();
    formData.append('query', query);
    
    if (filters && filters.length > 0) {
      filters.forEach(filter => {
        formData.append('filters', filter);
      });
    }
    
    // Always append limit, including when it's 0 (All)
    formData.append('limit', String(limit !== undefined ? limit : 10));
    
    return api.post('/api/search/text', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },
  
  // Multimodal Search
  async searchMultimodal({ file, query, weightImage = 0.5, filters, limit }: SearchMultimodalParams) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('query', query);
    formData.append('weight_image', String(weightImage));
    
    if (filters && filters.length > 0) {
      filters.forEach(filter => {
        formData.append('filters', filter);
      });
    }
    
    // Always append limit, including when it's 0 (All)
    formData.append('limit', String(limit !== undefined ? limit : 10));
    
    return api.post('/api/search/multimodal', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },
  
  // Get all images
  async getAllImages() {
    return api.get('/api/images');
  },
  
  // Get image by ID
  async getImageById(imageId: string) {
    return api.get(`/api/image/${imageId}`);
  },
  
  // Get all filters
  async getFilters() {
    return api.get('/api/filters');
  },
  
  // Add a filter
  async addFilter(filterQuery: string) {
    const formData = new FormData();
    formData.append('filter_query', filterQuery);
    
    return api.post('/api/filters', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },
  
  // Get filter progress
  async getFilterProgress(filterQuery: string) {
    return api.get(`/api/filter-progress?filter_query=${encodeURIComponent(filterQuery)}`);
  },
  
  // Delete a filter
  async deleteFilter(filterQuery: string) {
    return api.delete(`/api/filters/${encodeURIComponent(filterQuery)}`);
  },
  
  // Reset system
  async resetSystem() {
    return api.post('/api/reset');
  },
  
  // Update image metadata
  async updateMetadata({ imageId, description, customMetadata }: UpdateMetadataParams) {
    const formData = new FormData();
    formData.append('description', description);
    
    if (customMetadata) {
      formData.append('custom_metadata', customMetadata);
    }
    
    return api.put(`/api/metadata/${imageId}`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },
};

export default apiClient;