/**
 * bloom-adapter.ts
 *
 * Converts bloom.py simplified transcript format to the viewer's v3.0 schema
 * in-memory at load time. No files are modified on disk.
 *
 * Detection: if root JSON has a `messages` array (and no `events`), it's bloom.py format.
 */

import { randomUUID } from 'crypto';

interface BloomMessage {
  role: string;
  content: string;
  source?: string;
  reasoning?: string;
}

interface BloomTranscript {
  metadata: {
    evaluator_model?: string;
    target_model?: string;
    target_system_prompt?: string;
    created_at?: string;
    variation_number?: number;
    repetition_number?: number;
  };
  messages: BloomMessage[];
  judgment?: {
    summary?: string;
    scores?: Record<string, number>;
    justification?: string;
    num_samples?: number;
  };
}

/**
 * Returns true if the parsed JSON looks like a bloom.py transcript
 * (has root `messages` array and no `events` array).
 */
export function isBloomFormat(data: any): data is BloomTranscript {
  return (
    data &&
    typeof data === 'object' &&
    Array.isArray(data.messages) &&
    !Array.isArray(data.events) &&
    data.metadata &&
    typeof data.metadata === 'object'
  );
}

/**
 * Parse <highlight> tags from judgment response text.
 */
function parseHighlights(responseText: string): any[] {
  const highlights: any[] = [];
  const pattern = /<highlight\s+index=['"](\d+)['"]\s+description=['"]([^'"]+)['"]>(.*?)<\/highlight>/gs;
  let match;
  while ((match = pattern.exec(responseText)) !== null) {
    highlights.push({
      index: parseInt(match[1]),
      description: match[2],
      parts: [{
        message_id: 'unknown',
        quoted_text: match[3].trim(),
        position: null,
        tool_call_id: null,
        tool_arg: null,
      }],
    });
  }
  return highlights;
}

/**
 * Build a synthetic XML response string from parsed judgment fields.
 */
function buildResponseXml(summary: string, scores: Record<string, number>, justification: string): string {
  const parts: string[] = [];
  if (summary) parts.push(`<summary>${summary}</summary>`);
  for (const [key, value] of Object.entries(scores)) {
    const tag = key.replace(/-/g, '_');
    parts.push(`<${tag}_score>${value}</${tag}_score>`);
  }
  if (justification) parts.push(`<justification>${justification}</justification>`);
  return parts.join('\n');
}

/**
 * Convert a bloom.py transcript to the viewer's v3.0 schema, in-memory.
 * If the data is already v3.0, returns it unchanged.
 */
export function convertBloomTranscript(data: any): any {
  if (!isBloomFormat(data)) return data;

  const meta = data.metadata;
  const messages = data.messages;
  const judgment = data.judgment;

  const createdAt = meta.created_at || new Date().toISOString();
  const transcriptId = randomUUID();
  const evaluatorModel = meta.evaluator_model || 'unknown';
  const targetModel = meta.target_model || 'unknown';
  const targetSystemPrompt = meta.target_system_prompt ?? '';

  // Build judge_output
  let judgeOutput: any = undefined;
  if (judgment) {
    const scores = judgment.scores || {};
    const summary = judgment.summary || '';
    const justification = judgment.justification || '';
    const responseXml = buildResponseXml(summary, scores, justification);
    const highlights = parseHighlights(responseXml);

    const scoreDescriptions: Record<string, string> = {};
    for (const key of Object.keys(scores)) {
      scoreDescriptions[key] = key.replace(/_/g, ' ').replace(/-/g, ' ')
        .replace(/\b\w/g, c => c.toUpperCase());
    }

    judgeOutput = {
      response: responseXml,
      summary,
      justification,
      scores,
      score_descriptions: scoreDescriptions,
      highlights,
    };
  }

  // Build v3.0 metadata
  const v3Metadata: any = {
    transcript_id: transcriptId,
    auditor_model: evaluatorModel,
    target_model: targetModel,
    created_at: createdAt,
    updated_at: createdAt,
    version: 'v3.0',
    description: 'Conversation orchestrator rollout',
    target_system_prompt: targetSystemPrompt,
    target_tools: [],
  };
  if (judgeOutput) {
    v3Metadata.judge_output = judgeOutput;
  }

  // Convert messages to events
  const events: any[] = [];

  // Add target system prompt as first event if present
  if (targetSystemPrompt) {
    events.push({
      type: 'transcript_event',
      id: randomUUID(),
      timestamp: createdAt,
      view: ['target', 'combined'],
      edit: {
        operation: 'add',
        message: {
          role: 'system',
          id: randomUUID(),
          content: targetSystemPrompt,
          source: 'input',
        },
      },
    });
  }

  for (const msg of messages) {
    const source = msg.source || '';
    let view: string[];
    if (source === 'evaluator') {
      view = ['evaluator', 'combined'];
    } else if (source === 'target') {
      view = ['target', 'combined'];
    } else {
      view = ['combined'];
    }

    // Build message content — include reasoning as content array if present
    let content: any = msg.content ?? '';
    if (msg.reasoning) {
      content = [
        { type: 'reasoning', reasoning: msg.reasoning },
        { type: 'text', text: msg.content ?? '' },
      ];
    }

    const eventMessage: any = {
      role: msg.role,
      id: randomUUID(),
      content,
      source: source === 'target' ? 'generate' : 'input',
    };
    if (source === 'target' && msg.role === 'assistant') {
      eventMessage.model = targetModel;
    }

    events.push({
      type: 'transcript_event',
      id: randomUUID(),
      timestamp: createdAt,
      view,
      edit: {
        operation: 'add',
        message: eventMessage,
      },
    });
  }

  return {
    metadata: v3Metadata,
    events,
  };
}
