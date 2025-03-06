import { create } from 'zustand';
import { FilterState } from '@/types';
import apiClient from '@/lib/api';

const initialState: FilterState = {
  filters: [],
  isLoading: false,
  error: null,
};

export const useFilterStore = create<FilterState & {
  fetchFilters: () => Promise<void>;
  addFilter: (query: string) => Promise<void>;
  deleteFilter: (query: string) => Promise<void>;
}>((set) => ({
  ...initialState,
  
  fetchFilters: async () => {
    set({ isLoading: true, error: null });
    
    try {
      const response = await apiClient.getFilters();
      set({ filters: response.data.filters, isLoading: false });
    } catch (error) {
      console.error('Error fetching filters:', error);
      set({
        error: 'An error occurred while fetching filters. Please try again.',
        isLoading: false,
      });
    }
  },
  
  addFilter: async (query) => {
    if (!query.trim()) {
      set({ error: 'Please enter a filter query' });
      return;
    }
    
    set({ isLoading: true, error: null });
    
    try {
      const response = await apiClient.addFilter(query);
      set({ filters: response.data.filters, isLoading: false });
    } catch (error) {
      console.error('Error adding filter:', error);
      set({
        error: 'An error occurred while adding the filter. Please try again.',
        isLoading: false,
      });
    }
  },
  
  deleteFilter: async (query) => {
    set({ isLoading: true, error: null });
    
    try {
      const response = await apiClient.deleteFilter(query);
      set({ filters: response.data.filters, isLoading: false });
    } catch (error) {
      console.error('Error deleting filter:', error);
      set({
        error: 'An error occurred while deleting the filter. Please try again.',
        isLoading: false,
      });
    }
  },
})); 