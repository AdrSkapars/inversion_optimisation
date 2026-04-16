import type { VisibilityState, ColumnSizingState } from '@tanstack/svelte-table';
import { STORAGE_KEYS } from '$lib/shared/constants';

/**
 * Load column visibility from localStorage
 */
export function loadColumnVisibility(): VisibilityState {
  if (typeof window === 'undefined') return {};
  
  try {
    const stored = localStorage.getItem(STORAGE_KEYS.COLUMN_VISIBILITY);
    if (!stored) return {};
    
    const visibleColumns = JSON.parse(stored) as string[];
    const visibility: VisibilityState = {};
    
    // Set all columns to false first, then set visible ones to true
    visibleColumns.forEach(columnId => {
      visibility[columnId] = true;
    });
    
    return visibility;
  } catch (error) {
    console.warn('Failed to load column visibility from localStorage:', error);
    return {};
  }
}

/**
 * Save column visibility to localStorage
 */
export function saveColumnVisibility(columnVisibility: VisibilityState): void {
  if (typeof window === 'undefined') return;
  
  try {
    const visibleColumns = Object.entries(columnVisibility)
      .filter(([_, visible]) => visible)
      .map(([key, _]) => key);
    
    localStorage.setItem(STORAGE_KEYS.COLUMN_VISIBILITY, JSON.stringify(visibleColumns));
  } catch (error) {
    console.warn('Failed to save column visibility to localStorage:', error);
  }
}

/**
 * Load column sizing from localStorage
 */
export function loadColumnSizing(): ColumnSizingState {
  if (typeof window === 'undefined') return {};
  
  try {
    const stored = localStorage.getItem(STORAGE_KEYS.COLUMN_SIZING);
    return stored ? JSON.parse(stored) : {};
  } catch (error) {
    console.warn('Failed to load column sizing from localStorage:', error);
    return {};
  }
}

/**
 * Save column sizing to localStorage
 */
export function saveColumnSizing(columnSizing: ColumnSizingState): void {
  if (typeof window === 'undefined') return;
  
  try {
    localStorage.setItem(STORAGE_KEYS.COLUMN_SIZING, JSON.stringify(columnSizing));
  } catch (error) {
    console.warn('Failed to save column sizing to localStorage:', error);
  }
}