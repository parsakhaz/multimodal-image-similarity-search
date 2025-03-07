// Image metadata type
export interface ImageMetadata {
  id: string;
  filename: string;
  description: string;
  custom_metadata?: string;
  url: string;
  thumbnail_url: string;
  processed_url?: string;
  created_at: string;
  similarity_score?: number;
}

// Search result type
export interface SearchResult {
  results: ImageMetadata[];
}

// Filter type
export interface Filter {
  query: string;
  display?: string;
}

// Response types
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

export interface FiltersResponse {
  filters: string[];
}

export interface ImagesResponse {
  images: ImageMetadata[];
}

// Search types
export type SearchType = 'image' | 'text' | 'multimodal';

export interface SearchState {
  searchType: SearchType;
  query: string;
  selectedImage: File | null;
  weightImage: number;
  filters: string[];
  results: ImageMetadata[];
  isLoading: boolean;
  error: string | null;
  resultLimit: number;
}

// Upload types
export interface UploadState {
  file: File | null;
  description: string;
  customMetadata: string;
  removeBg: boolean;
  isUploading: boolean;
  error: string | null;
  success: boolean;
}

// Filter manager state
export interface FilterState {
  filters: string[];
  isLoading: boolean;
  error: string | null;
}

// Image manager state
export interface ImageManagerState {
  images: ImageMetadata[];
  selectedImage: ImageMetadata | null;
  isLoading: boolean;
  error: string | null;
} 