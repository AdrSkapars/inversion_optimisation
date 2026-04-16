import { promises as fs } from 'fs';
import path from 'path';
import type { ConfigIndex, TranscriptIndexEntry, SuiteIndex, AggregatedIndex } from './index-types';

// Server-side check
if (typeof window !== 'undefined') {
  throw new Error('index-builder can only be used on the server side');
}

const INDEX_VERSION = '1.0';

/**
 * Scan a directory for all config folders (folders containing judgment.json)
 */
async function findConfigFolders(transcriptDir: string, basePath: string = ''): Promise<string[]> {
  const configs: string[] = [];
  const currentPath = path.join(transcriptDir, basePath);

  try {
    const entries = await fs.readdir(currentPath, { withFileTypes: true });

    const hasJudgment = entries.some(e => e.isFile() && e.name === 'judgment.json');
    if (hasJudgment) {
      configs.push(basePath);
      return configs; // Don't recurse into config folders
    }

    for (const entry of entries) {
      if (entry.isDirectory() && !entry.name.startsWith('.')) {
        const subPath = basePath ? path.join(basePath, entry.name) : entry.name;
        const subConfigs = await findConfigFolders(transcriptDir, subPath);
        configs.push(...subConfigs);
      }
    }
  } catch (error) {
    console.error(`Error scanning ${currentPath}:`, error);
  }

  return configs;
}

/**
 * Build index for a single config directory by reading judgment.json + evaluation.json + cfg.json
 */
async function buildConfigIndex(transcriptDir: string, configPath: string): Promise<ConfigIndex | null> {
  try {
    const fullConfigPath = path.resolve(transcriptDir, configPath);

    // Read judgment.json — required
    let judgmentData: any;
    try {
      const content = await fs.readFile(path.join(fullConfigPath, 'judgment.json'), 'utf-8');
      judgmentData = JSON.parse(content);
    } catch {
      console.warn(`⚠️ [INDEX-BUILDER] No judgment.json found for ${configPath}`);
      return null;
    }

    // Read evaluation.json — optional
    let evaluationMetadata: any = {};
    try {
      const content = await fs.readFile(path.join(fullConfigPath, 'evaluation.json'), 'utf-8');
      evaluationMetadata = JSON.parse(content).metadata || {};
    } catch { /* optional */ }

    // Read cfg.json for auditor/target model names — optional
    let auditorModel: string | undefined;
    let targetModel: string | undefined;
    try {
      const content = await fs.readFile(path.join(fullConfigPath, 'cfg.json'), 'utf-8');
      const cfgData = JSON.parse(content);
      auditorModel = cfgData?.rollout?.evaluator || cfgData?.rollout?.model;
      targetModel = cfgData?.rollout?.target;
    } catch { /* optional */ }

    // Build transcript entries from judgment.json's judgments array
    const judgments: any[] = judgmentData.judgments || [];
    const skipKeys = new Set(['variation_number', 'variation_description', 'repetition_number',
      'behavior_presence', 'justification', 'summary', 'full_judgment_response',
      'num_samples', 'individual_samples']);

    const transcripts: TranscriptIndexEntry[] = judgments.map((j: any) => {
      const v = j.variation_number ?? 0;
      const r = j.repetition_number ?? 1;
      const filename = `transcript_v${v}r${r}.json`;
      const normalizedConfigPath = configPath.replace(/\\/g, '/');
      const filePath = normalizedConfigPath
        ? `${normalizedConfigPath}/transcripts/${filename}`
        : `transcripts/${filename}`;

      const scores: Record<string, number> = {};
      if (j.behavior_presence != null) scores['behavior_presence'] = j.behavior_presence;
      for (const [k, val] of Object.entries(j)) {
        if (!skipKeys.has(k) && typeof val === 'number') {
          scores[k] = val as number;
        }
      }

      const summary = j.summary || '';
      return {
        id: filename.replace('.json', ''),
        transcript_id: `${normalizedConfigPath}/transcript_v${v}r${r}`,
        _filePath: filePath,
        summary: summary.length > 200 ? summary.substring(0, 200) + '...' : summary,
        scores,
      };
    });

    const pathParts = configPath.split(/[/\\]/);
    const configName = pathParts[pathParts.length - 1] || path.basename(transcriptDir);
    const normalizedPath = configPath.replace(/\\/g, '/') || configName;

    return {
      version: INDEX_VERSION,
      generated_at: new Date().toISOString(),
      config: {
        name: configName,
        path: normalizedPath,
        auditor_model: auditorModel,
        target_model: targetModel,
        transcript_count: transcripts.length,
      },
      summary_statistics: judgmentData.summary_statistics,
      metajudge: {
        response: judgmentData.metajudgment_response,
        justification: judgmentData.metajudgment_justification,
      },
      evaluation_metadata: evaluationMetadata,
      transcripts,
    };
  } catch (error) {
    console.error(`❌ [INDEX-BUILDER] Failed to build index for ${configPath}:`, error);
    return null;
  }
}

/**
 * Load all indexes by scanning the directory and reading judgment.json files directly
 */
export async function loadAggregatedIndexes(transcriptDir: string): Promise<AggregatedIndex> {
  console.log(`📊 [INDEX-BUILDER] Scanning: ${transcriptDir}`);

  const configPaths = await findConfigFolders(transcriptDir);
  console.log(`📁 [INDEX-BUILDER] Found ${configPaths.length} config folders`);

  const configIndexes: ConfigIndex[] = [];
  for (const configPath of configPaths) {
    const index = await buildConfigIndex(transcriptDir, configPath);
    if (index) configIndexes.push(index);
  }

  // Group configs by suite (first path component)
  const suiteMap = new Map<string, ConfigIndex[]>();
  for (const config of configIndexes) {
    const suiteName = config.config.path.split('/')[0];
    if (!suiteMap.has(suiteName)) suiteMap.set(suiteName, []);
    suiteMap.get(suiteName)!.push(config);
  }

  const suites: SuiteIndex[] = Array.from(suiteMap.entries()).map(([name, configs]) => ({
    name,
    path: name,
    configs,
  }));

  console.log(`✅ [INDEX-BUILDER] Loaded ${configIndexes.length} configs in ${suites.length} suites`);

  return {
    version: INDEX_VERSION,
    generated_at: new Date().toISOString(),
    suites,
  };
}
