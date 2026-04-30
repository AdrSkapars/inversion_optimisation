import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import path from 'path';
import { TRANSCRIPT_DIR } from '$lib/server/config';
import { loadAggregatedIndexes } from '$lib/server/cache/index-builder';

export const GET: RequestHandler = async ({ url }) => {
  const rootDir = url.searchParams.get('rootDir');
  const transcriptDir = rootDir
    ? path.join(TRANSCRIPT_DIR, rootDir)
    : TRANSCRIPT_DIR;

  try {
    const index = await loadAggregatedIndexes(transcriptDir);
    return json(index);
  } catch (error: any) {
    return json({ error: error.message }, { status: 500 });
  }
};
