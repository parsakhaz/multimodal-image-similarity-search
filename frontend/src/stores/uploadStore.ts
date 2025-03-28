import { create } from 'zustand';
import { UploadState } from '@/types';
import apiClient from '@/lib/api';

const initialState: UploadState = {
  file: null,
  description: '',
  customMetadata: '',
  removeBg: false,
  isUploading: false,
  error: null,
  success: false,
};

export const useUploadStore = create<UploadState & {
  setFile: (file: File | null) => void;
  setDescription: (description: string) => void;
  setCustomMetadata: (metadata: string) => void;
  setRemoveBg: (remove: boolean) => void;
  resetForm: () => void;
  uploadImage: () => Promise<void>;
}>((set, get) => ({
  ...initialState,
  
  setFile: (file) => set({ file, success: false, error: null }),
  
  setDescription: (description) => set({ description }),
  
  setCustomMetadata: (customMetadata) => set({ customMetadata }),
  
  setRemoveBg: (removeBg) => set({ removeBg }),
  
  resetForm: () => set(initialState),
  
  uploadImage: async () => {
    const { file, description, customMetadata, removeBg } = get();
    
    if (!file) {
      set({ error: 'Please select an image to upload' });
      return;
    }
    
    set({ isUploading: true, error: null, success: false });
    
    try {
      await apiClient.uploadImage({
        file,
        description,
        customMetadata,
        removeBg,
      });
      
      set({
        isUploading: false,
        success: true,
        file: null,
        description: '',
        customMetadata: '',
      });
    } catch (error: unknown) {
      console.error('Upload error:', error);
      
      // Check if it's a duplicate image error (409 Conflict)
      const axiosError = error as {
        response?: {
          status?: number;
          data?: {
            error?: string;
            metadata?: {
              id: string;
              filename?: string;
              description?: string;
              url?: string;
              [key: string]: string | number | boolean | undefined;
            };
            message?: string;
          }
        }
      };
      
      if (axiosError.response?.status === 409 && axiosError.response?.data?.error?.includes('Duplicate')) {
        set({
          error: `This image already exists in the database (ID: ${axiosError.response.data.metadata?.id.substring(0, 8)}...)`,
          isUploading: false,
          success: false,
        });
      } else {
        set({
          error: axiosError.response?.data?.error || 'An error occurred while uploading the image. Please try again.',
          isUploading: false,
          success: false,
        });
      }
    }
  },
})); 