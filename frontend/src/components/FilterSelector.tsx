import React, { useEffect } from 'react';
import { useFilterStore } from '@/stores/filterStore';

interface FilterSelectorProps {
  selectedFilters: string[];
  onFilterChange: (filters: string[]) => void;
  className?: string;
}

const FilterSelector: React.FC<FilterSelectorProps> = ({
  selectedFilters,
  onFilterChange,
  className = '',
}) => {
  const { filters, isLoading, error, fetchFilters } = useFilterStore();

  useEffect(() => {
    fetchFilters();
  }, [fetchFilters]);

  const handleFilterToggle = (filter: string) => {
    if (selectedFilters.includes(filter)) {
      onFilterChange(selectedFilters.filter(f => f !== filter));
    } else {
      onFilterChange([...selectedFilters, filter]);
    }
  };

  if (isLoading) {
    return <div className="text-sm text-gray-500">Loading filters...</div>;
  }

  if (error) {
    return <div className="text-sm text-red-500">{error}</div>;
  }

  if (filters.length === 0) {
    return <div className="text-sm text-gray-500">No filters available</div>;
  }

  return (
    <div className={`space-y-2 ${className}`}>
      <h3 className="text-sm font-medium">Filters</h3>
      
      <div className="space-y-1">
        {filters.map((filter) => (
          <div key={filter} className="flex items-center">
            <input
              type="checkbox"
              id={`filter-${filter}`}
              checked={selectedFilters.includes(filter)}
              onChange={() => handleFilterToggle(filter)}
              className="h-4 w-4 text-blue-600 rounded border-gray-300 focus:ring-blue-500"
            />
            <label
              htmlFor={`filter-${filter}`}
              className="ml-2 text-sm text-gray-700"
            >
              {filter}
            </label>
          </div>
        ))}
      </div>
    </div>
  );
};

export default FilterSelector; 