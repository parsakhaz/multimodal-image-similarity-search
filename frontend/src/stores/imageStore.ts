import { create } from 'zustand';
import { ImageManagerState, ImageMetadata } from '@/types';
import apiClient from '@/lib/api';

const initialState: ImageManagerState = {
  images: [],
  selectedImage: null,
  isLoading: false,
  error: null,
};

export const useImageStore = create<ImageManagerState & {
  fetchImages: () => Promise<void>;
  fetchImageById: (imageId: string) => Promise<void>;
  selectImage: (image: ImageMetadata | null) => void;
  updateMetadata: (imageId: string, description: string, customMetadata?: string) => Promise<void>;
}>((set, get) => ({
  ...initialState,
  
  fetchImages: async () => {
    set({ isLoading: true, error: null });
    
    try {
      const response = await apiClient.getAllImages();
      set({ images: response.data.images, isLoading: false });
    } catch (error) {
      console.error('Error fetching images:', error);
      set({
        error: 'An error occurred while fetching images. Please try again.',
        isLoading: false,
      });
    }
  },
  
  fetchImageById: async (imageId) => {
    set({ isLoading: true, error: null });
    
    try {
      const response = await apiClient.getImageById(imageId);
      if (response.data.success && response.data.image) {
        // Update the image in the images array if it exists
        const { images } = get();
        const existingIndex = images.findIndex(img => img.id === imageId);
        
        if (existingIndex >= 0) {
          const updatedImages = [...images];
          updatedImages[existingIndex] = response.data.image;
          set({ 
            images: updatedImages,
            selectedImage: response.data.image,
            isLoading: false 
          });
        } else {
          // If not in the array, just set the selected image
          set({ 
            selectedImage: response.data.image,
            isLoading: false 
          });
        }
      } else {
        set({ isLoading: false });
      }
    } catch (error) {
      console.error('Error fetching image by ID:', error);
      set({
        error: 'An error occurred while fetching the image. Please try again.',
        isLoading: false,
      });
    }
  },
  
  selectImage: (image) => set({ selectedImage: image }),
  
  updateMetadata: async (imageId, description, customMetadata) => {
    set({ isLoading: true, error: null });
    
    try {
      const response = await apiClient.updateMetadata({
        imageId,
        description,
        customMetadata,
      });
      
      // Update the image in the local state
      const { images } = get();
      const updatedImages = images.map(img => 
        img.id === imageId ? { ...img, description, custom_metadata: customMetadata } : img
      );
      
      set({ 
        images: updatedImages, 
        isLoading: false,
        selectedImage: response.data.metadata,
      });
    } catch (error) {
      console.error('Error updating metadata:', error);
      set({
        error: 'An error occurred while updating the metadata. Please try again.',
        isLoading: false,
      });
    }
  },
})); 