import { promises as fs } from 'fs';
import path from 'path';
import type { TranscriptDisplayFull, TranscriptMetadata, Transcript } from '$lib/shared/types';
import { validateTranscriptData, type ValidationResult, formatValidationErrors } from '$lib/shared/schema-validator';
import { createTranscriptDisplay, extractTranscriptMetadata } from '$lib/shared/utils/transcript-utils';
import { convertBloomTranscript } from '$lib/server/data-loading/bloom-adapter';

// This module only works on the server side
if (typeof window !== 'undefined') {
  throw new Error('transcript-loader can only be used on the server side');
}

/**
 * Load a complete transcript from a file path
 */
export async function loadTranscriptFromFile(filePath: string): Promise<TranscriptDisplayFull | null> {
  try {
    const content = await fs.readFile(filePath, 'utf-8');

    // Parse, convert bloom.py format if needed, then validate
    const parsed = convertBloomTranscript(JSON.parse(content));
    const validationResult = validateTranscriptData(parsed);

    if (!validationResult.valid) {
      const details = formatValidationErrors(validationResult.errors);
      const err = new Error(`Invalid transcript file: ${filePath}\n${details}`);
      (err as any).code = 'TRANSCRIPT_VALIDATION_FAILED';
      (err as any).validation = validationResult;
      throw err;
    }
    
    const transcript = validationResult.data as Transcript;
    
    // Ensure transcript_id exists in metadata
    if (!transcript.metadata?.transcript_id) {
      throw new Error(`Missing transcript_id in metadata for file: ${filePath}`);
    }
    
    return createTranscriptDisplay(transcript, filePath);
    
  } catch (error: any) {
    if (error.code === 'ENOENT') {
      return null; // File not found - expected case
    }
    throw new Error(`Failed to load transcript from ${filePath}: ${error.message}`);
  }
}

/**
 * Load only metadata from a transcript file
 */
export async function loadMetadataFromFile(filePath: string): Promise<TranscriptMetadata | null> {
  try {
    const content = await fs.readFile(filePath, 'utf-8');
    const data = convertBloomTranscript(JSON.parse(content));

    if (!data?.metadata) {
      throw new Error(`Invalid transcript structure in ${filePath}: missing metadata`);
    }

    return data.metadata as TranscriptMetadata;
    
  } catch (error: any) {
    if (error.code === 'ENOENT') {
      return null; // File not found - expected case
    }
    
    if (error instanceof SyntaxError) {
      throw new Error(`Invalid JSON in ${filePath}: ${error.message}`);
    }
    
    throw new Error(`Failed to load metadata from ${filePath}: ${error.message}`);
  }
}

/**
 * Validate a transcript file
 */
export async function validateTranscriptFile(filePath: string): Promise<ValidationResult> {
  try {
    const content = await fs.readFile(filePath, 'utf-8');
    const parsed = convertBloomTranscript(JSON.parse(content));
    return validateTranscriptData(parsed);
    
  } catch (error: any) {
    return {
      isValid: false,
      valid: false,
      errors: [{
        message: error.code === 'ENOENT'
          ? `File not found: ${filePath}`
          : `Failed to read file: ${error.message}`,
        path: 'file',
        value: filePath,
        schema: null
      }],
      data: null
    };
  }
}