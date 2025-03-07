import { create } from 'zustand';
import { SearchState, SearchType } from '@/types';
import apiClient from '@/lib/api';

const initialState: SearchState = {
  searchType: 'text',
  query: '',
  selectedImage: null,
  weightImage: 0.5,
  filters: [],
  results: [],
  isLoading: false,
  error: null,
  resultLimit: 10, // Default to 10 results
};

export const useSearchStore = create<SearchState & {
  setSearchType: (type: SearchType) => void;
  setQuery: (query: string) => void;
  setSelectedImage: (file: File | null) => void;
  setWeightImage: (weight: number) => void;
  setFilters: (filters: string[]) => void;
  setResultLimit: (limit: number) => void;
  clearResults: () => void;
  searchByText: () => Promise<void>;
  searchByImage: () => Promise<void>;
  searchMultimodal: () => Promise<void>;
}>((set, get) => ({
  ...initialState,
  
  setSearchType: (type) => set({ searchType: type }),
  
  setQuery: (query) => set({ query }),
  
  setSelectedImage: (file) => set({ selectedImage: file }),
  
  setWeightImage: (weight) => set({ weightImage: weight }),
  
  setFilters: (filters) => set({ filters }),
  
  setResultLimit: (limit) => set({ resultLimit: limit }),
  
  clearResults: () => set({ results: [], error: null }),
  
  searchByText: async () => {
    const { query, filters, resultLimit } = get();
    
    if (!query.trim() && (!filters || filters.length === 0)) {
      set({ error: 'Please enter a search query or select at least one filter' });
      return;
    }
    
    set({ isLoading: true, error: null });
    
    try {
      const response = await apiClient.searchByText({ 
        query, 
        filters,
        limit: resultLimit
      });
      set({ results: response.data.results, isLoading: false });
    } catch (error) {
      console.error('Text search error:', error);
      set({ 
        error: 'An error occurred while searching. Please try again.', 
        isLoading: false 
      });
    }
  },
  
  searchByImage: async () => {
    const { selectedImage, filters, resultLimit } = get();
    
    if (!selectedImage) {
      set({ error: 'Please select an image to search' });
      return;
    }
    
    set({ isLoading: true, error: null });
    
    try {
      const response = await apiClient.searchByImage({ 
        file: selectedImage, 
        filters,
        limit: resultLimit
      });
      set({ results: response.data.results, isLoading: false });
    } catch (error) {
      console.error('Image search error:', error);
      set({ 
        error: 'An error occurred while searching. Please try again.', 
        isLoading: false 
      });
    }
  },
  
  searchMultimodal: async () => {
    const { query, selectedImage, weightImage, filters, resultLimit } = get();
    
    if (!selectedImage) {
      set({ error: 'Please select an image to search' });
      return;
    }
    
    if (!query.trim()) {
      set({ error: 'Please enter a search query' });
      return;
    }
    
    set({ isLoading: true, error: null });
    
    try {
      const response = await apiClient.searchMultimodal({ 
        file: selectedImage, 
        query,
        weightImage,
        filters,
        limit: resultLimit
      });
      set({ results: response.data.results, isLoading: false });
    } catch (error) {
      console.error('Multimodal search error:', error);
      set({ 
        error: 'An error occurred while searching. Please try again.', 
        isLoading: false 
      });
    }
  },
})); 