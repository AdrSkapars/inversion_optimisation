import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { promises as fs } from 'fs';
import path from 'path';
import { TRANSCRIPT_DIR } from '$lib/server/config';
import { loadTranscriptFromFile, loadMetadataFromFile } from '$lib/server/core/transcript-loader';

export const GET: RequestHandler = async ({ url }) => {
  const filePath = url.searchParams.get('filePath');
  const metadataOnly = url.searchParams.get('metadataOnly') === 'true';

  if (!filePath) {
    return json({ success: false, error: 'Missing filePath parameter' }, { status: 400 });
  }

  const fullPath = path.join(TRANSCRIPT_DIR, filePath);

  try {
    if (metadataOnly) {
      const metadata = await loadMetadataFromFile(fullPath);
      if (!metadata) return json({ success: false, error: 'Not found' }, { status: 404 });
      return json({ success: true, data: metadata });
    }

    const transcript = await loadTranscriptFromFile(fullPath);
    if (!transcript) return json({ success: false, error: 'Not found' }, { status: 404 });

    // Attach scenario description from ideation.json when available.
    // Expected path: .../round_N/transcripts/transcript_vXrY.json
    //             → .../round_N/ideation.json
    try {
      const transcriptsDir = path.dirname(fullPath);
      const roundDir = path.dirname(transcriptsDir);
      const ideationPath = path.join(roundDir, 'ideation.json');
      const ideationContent = await fs.readFile(ideationPath, 'utf-8');
      const ideation = JSON.parse(ideationContent);

      const filename = path.basename(fullPath);
      const match = filename.match(/transcript_v(\d+)r\d+\.json/);
      if (match) {
        const variationIndex = parseInt(match[1]) - 1; // variation numbers are 1-indexed
        const variations: any[] = ideation.variations || [];
        if (variationIndex >= 0 && variationIndex < variations.length) {
          const scenarioDescription = variations[variationIndex]?.description ?? null;
          if (scenarioDescription && 'transcript' in transcript) {
            (transcript as any).transcript.metadata.scenario_description = scenarioDescription;
          }
        }
      }
    } catch { /* ideation.json absent — ok */ }

    return json({ success: true, data: transcript });
  } catch (error: any) {
    if (error.code === 'ENOENT') {
      return json({ success: false, error: 'Transcript not found' }, { status: 404 });
    }
    return json({ success: false, error: error.message }, { status: 500 });
  }
};
