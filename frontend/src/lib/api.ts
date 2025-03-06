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
}

export interface SearchTextParams {
  query: string;
  filters?: string[];
}

export interface SearchMultimodalParams {
  file: File;
  query: string;
  weightImage?: number;
  filters?: string[];
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
  
  // Folder Upload
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
  async searchByImage({ file, filters }: SearchImageParams) {
    const formData = new FormData();
    formData.append('file', file);
    
    if (filters && filters.length > 0) {
      filters.forEach(filter => {
        formData.append('filters', filter);
      });
    }
    
    return api.post('/api/search/image', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },
  
  // Text Search
  async searchByText({ query, filters }: SearchTextParams) {
    let url = `/api/search/text?query=${encodeURIComponent(query)}`;
    
    if (filters && filters.length > 0) {
      filters.forEach(filter => {
        url += `&filters=${encodeURIComponent(filter)}`;
      });
    }
    
    return api.get(url);
  },
  
  // Multimodal Search
  async searchMultimodal({ file, query, weightImage = 0.5, filters }: SearchMultimodalParams) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('query', query);
    formData.append('weight_image', String(weightImage));
    
    if (filters && filters.length > 0) {
      filters.forEach(filter => {
        formData.append('filters', filter);
      });
    }
    
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