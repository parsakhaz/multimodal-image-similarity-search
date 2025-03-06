import { create } from 'zustand';
import { SearchState, SearchType, ImageMetadata } from '@/types';
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
};

export const useSearchStore = create<SearchState & {
  setSearchType: (type: SearchType) => void;
  setQuery: (query: string) => void;
  setSelectedImage: (file: File | null) => void;
  setWeightImage: (weight: number) => void;
  setFilters: (filters: string[]) => void;
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
  
  clearResults: () => set({ results: [], error: null }),
  
  searchByText: async () => {
    const { query, filters } = get();
    
    if (!query.trim()) {
      set({ error: 'Please enter a search query' });
      return;
    }
    
    set({ isLoading: true, error: null });
    
    try {
      const response = await apiClient.searchByText({ query, filters });
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
    const { selectedImage, filters } = get();
    
    if (!selectedImage) {
      set({ error: 'Please select an image to search' });
      return;
    }
    
    set({ isLoading: true, error: null });
    
    try {
      const response = await apiClient.searchByImage({ 
        file: selectedImage, 
        filters 
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
    const { query, selectedImage, weightImage, filters } = get();
    
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
        filters 
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