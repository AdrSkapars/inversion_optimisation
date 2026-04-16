<script lang="ts">
  import type { Message, SystemMessage, UserMessage, AssistantMessage, ToolMessage, APIFailureMessage, InfoMessage, ToolCall } from '$lib/shared/types';
  import type { CopyAction } from '$lib/client/utils/copy-utils';
  import { getMessageBorderColor, getMessageBackgroundColor } from '$lib/shared/transcript-parser';
  import { JsonViewer } from '@kaifronsdal/svelte-json-viewer';
  import { themeString } from '$lib/shared/theme';
  import markdownit from 'markdown-it';
  import hljs from 'highlight.js/lib/core';
  import python from 'highlight.js/lib/languages/python';
  import bash from 'highlight.js/lib/languages/bash';
  import plaintext from 'highlight.js/lib/languages/plaintext';

  hljs.registerLanguage('python', python);
  hljs.registerLanguage('bash', bash);
  hljs.registerLanguage('plaintext', plaintext);

  interface Props {
    message: Message;
    messageIndex: number;
    columnIndex: number;
    isOpen: boolean;
    isVisible?: boolean;
    onToggle: (messageId: string) => void;
    onCopy: (action: CopyAction) => void;
    highlightText?: string | null;
  }


  let {
    message,
    messageIndex,
    columnIndex,
    isOpen,
    isVisible = true,
    onToggle,
    onCopy,
    highlightText = null
  }: Props = $props();

  // Proper text extraction from original code
  function extractTextFromContent(content: string | Array<string | Record<string, any>>): string | null {
    if (typeof content === 'string') {
      return content;
    }

    if (Array.isArray(content)) {
      const textParts: string[] = [];
      for (const item of content) {
        if (typeof item === 'string') {
          textParts.push(item);
        } else if (typeof item === 'object' && item !== null && item.type === 'text') {
          textParts.push(item.text || '');
        }
      }
      return textParts.join('');
    }

    return null;
  }

  // Hash function to generate consistent colors for message names
  function hashStringToColor(str: string): string {
    if (!str) return 'badge-neutral';
    
    // Simple hash function
    let hash = 42;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    
    // Map hash to predefined badge colors
    const colors = [
      'badge-primary',
      'badge-secondary', 
      'badge-accent',
      'badge-info',
      'badge-success',
      'badge-warning'
    ];
    
    const colorIndex = Math.abs(hash) % colors.length;
    return colors[colorIndex];
  }

  function toTitleCase(str: string): string {
    if (!str) return '';
    return str.replace(/\w\S*/g, (txt) => 
      txt.charAt(0).toUpperCase() + txt.slice(1).toLowerCase()
    );
  }

  const md = markdownit({
    html: true,
    linkify: true,
    breaks: true,
    highlight: function (str: string, lang: string) {
      if (lang && hljs.getLanguage(lang)) {
        try {
          return hljs.highlight(str, { language: lang, ignoreIllegals: true }).value;
        } catch {}
      }
      return '';
    }
  });

  function renderMarkdown(content: string): string {
    try {
      let result = md.render(content || '');

      // If we have highlightText and it exists in the content, highlight it
      if (highlightText && result.includes(highlightText)) {
        // Escape special regex characters
        const escapedHighlight = highlightText.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const regex = new RegExp(`(${escapedHighlight})`, 'gi');
        result = result.replace(regex, '<mark class="quote-highlight px-1 rounded">$1</mark>');
      }

      return result;
    } catch {
      return content || '';
    }
  }

  // Helper function to highlight text in plain content (non-markdown)
  function highlightPlainText(text: string): string {
    if (!highlightText || !text.includes(highlightText)) {
      return text;
    }

    // Escape special regex characters and HTML-escape the text
    const escapedHighlight = highlightText.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const regex = new RegExp(`(${escapedHighlight})`, 'gi');

    // HTML-escape the text first to prevent XSS
    const escaped = text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');

    // Apply highlight with pale yellow for dark mode to keep white text readable
    const highlighted = escaped.replace(regex, '<mark class="quote-highlight px-1 rounded">$1</mark>');

    return highlighted;
  }

  // Left-edge accent mapped from badge color for assistant messages
  function leftEdgeClassForAssistant(message: Message): string {
    if ((message as any).type !== 'assistant') return '';
    const name = getMessageSourceLabel(message) || '';
    const badge = hashStringToColor(name);
    const badgeToBorder: Record<string, string> = {
      // DaisyUI defaults: primary≈blue, secondary≈violet, accent≈pink, info≈sky, success≈emerald/green, warning≈amber, neutral≈zinc
      'badge-primary': 'border-l border-blue-500',
      'badge-secondary': 'border-l border-violet-500',
      'badge-accent': 'border-l border-pink-500',
      'badge-info': 'border-l border-sky-500',
      'badge-success': 'border-l border-emerald-500',
      'badge-warning': 'border-l border-amber-500',
      'badge-neutral': 'border-l border-zinc-500'
    };
    return badgeToBorder[badge] || 'border-l border-zinc-500';
  }

  // Minimal YAML serializer for displaying tool arguments
  function toYaml(value: any, indent: number = 0): string {
    const pad = (n: number) => '  '.repeat(n);
    const isPlainObject = (v: any) => Object.prototype.toString.call(v) === '[object Object]';

    if (value === null || value === undefined) return 'null';
    if (typeof value === 'string') {
      // Use JSON.parse to properly unescape the string (like Python's print() would show)
      let unescaped: string;
      try {
        // Wrap in quotes and parse as JSON to handle all escape sequences automatically
        unescaped = JSON.parse('"' + value.replace(/"/g, '\\"') + '"');
      } catch {
        // If JSON.parse fails (malformed escapes), fall back to the original string
        unescaped = value;
      }
      
      // Check if the unescaped string contains newlines
      if (unescaped.includes('\n')) {
        const lines = unescaped.split('\n');
        // Use literal block scalar (|) to preserve newlines and formatting
        return lines.map(line => pad(indent + 1) + line).join('\n');
      }
      
      // For single-line strings, return the unescaped version directly (no quotes needed in YAML)
      // But we need to be careful with special YAML characters
      const needsQuotes = /^[\s]*$|^[>|]|[:#\[\]{}*&!%@`]/.test(unescaped) || 
                         unescaped.startsWith('-') || 
                         unescaped.match(/^\d/) ||
                         ['true', 'false', 'null', 'yes', 'no', 'on', 'off'].includes(unescaped.toLowerCase());
      
      return needsQuotes ? JSON.stringify(unescaped) : unescaped;
    }
    if (typeof value === 'number' || typeof value === 'boolean') return String(value);
    if (Array.isArray(value)) {
      if (value.length === 0) return '[]';
      return value
        .map((item) => {
          const rendered = toYaml(item, indent + 1);
          if (rendered.includes('\n')) {
            const lines = rendered.split('\n').map((l) => pad(indent + 1) + l).join('\n');
            return `${pad(indent)}-\n${lines}`;
          }
          return `${pad(indent)}- ${rendered}`;
        })
        .join('\n');
    }
    if (isPlainObject(value)) {
      const entries = Object.entries(value);
      if (entries.length === 0) return '{}';
      return entries
        .map(([k, v]) => {
          const rendered = toYaml(v, indent + 1);
          if (rendered.includes('\n')) {
            const lines = rendered.split('\n').map((l) => pad(indent + 1) + l).join('\n');
            return `${pad(indent)}${k}:\n${lines}`;
          }
          return `${pad(indent)}${k}: ${rendered}`;
        })
        .join('\n');
    }
    // Fallback for other types
    try { return JSON.stringify(value); } catch { return String(value); }
  }

  // Derive a human-readable source label for the message (schema v3)
  function getMessageSourceLabel(message: Message): string | null {
    const meta: any = (message as any).metadata;
    if (meta && typeof meta.source === 'string' && meta.source) {
      return meta.source;
    }
    if ('name' in (message as any) && typeof (message as any).name === 'string') {
      return (message as any).name;
    }
    if (message.type === 'api_failure') return 'API Error';
    if (message.type === 'info') return 'Info';
    return null;
  }

  // Collapsible state for tool calls within assistant messages
  let collapsedToolCallById: Record<string, boolean> = $state({});
  function isToolCallCollapsed(id: string): boolean {
    return !!collapsedToolCallById[id];
  }
  function toggleToolCallCollapsed(id: string) {
    collapsedToolCallById[id] = !collapsedToolCallById[id];
  }

  // Raw JSON view toggle state
  let showRawJson = $state(false);
  function toggleRawJsonView() {
    showRawJson = !showRawJson;
  }

  // Color coding based on message type
  function getMessageColors(type: string): { bg: string; headerBg: string; border: string; badgeBg: string; badgeText: string } {
    switch (type) {
      case 'tool':
        return {
          bg: 'bg-blue-50/50 dark:bg-blue-950/20',
          headerBg: 'bg-blue-100 dark:bg-blue-900/40',
          border: 'border-blue-200 dark:border-blue-800',
          badgeBg: 'bg-blue-500 dark:bg-blue-600',
          badgeText: 'text-white'
        };
      case 'assistant':
        return {
          bg: 'bg-purple-50/50 dark:bg-purple-950/20',
          headerBg: 'bg-purple-100 dark:bg-purple-900/40',
          border: 'border-purple-200 dark:border-purple-800',
          badgeBg: 'bg-purple-500 dark:bg-purple-600',
          badgeText: 'text-white'
        };
      case 'user':
        return {
          bg: 'bg-green-50/50 dark:bg-green-950/20',
          headerBg: 'bg-green-100 dark:bg-green-900/40',
          border: 'border-green-200 dark:border-green-800',
          badgeBg: 'bg-green-500 dark:bg-green-600',
          badgeText: 'text-white'
        };
      case 'system':
        return {
          bg: 'bg-amber-50/50 dark:bg-amber-950/20',
          headerBg: 'bg-amber-100 dark:bg-amber-900/40',
          border: 'border-amber-200 dark:border-amber-800',
          badgeBg: 'bg-amber-500 dark:bg-amber-600',
          badgeText: 'text-white'
        };
      case 'api_failure':
        return {
          bg: 'bg-red-50/50 dark:bg-red-950/20',
          headerBg: 'bg-red-100 dark:bg-red-900/40',
          border: 'border-red-200 dark:border-red-800',
          badgeBg: 'bg-red-500 dark:bg-red-600',
          badgeText: 'text-white'
        };
      case 'info':
        return {
          bg: 'bg-slate-50/50 dark:bg-slate-950/20',
          headerBg: 'bg-slate-100 dark:bg-slate-900/40',
          border: 'border-slate-200 dark:border-slate-800',
          badgeBg: 'bg-slate-500 dark:bg-slate-600',
          badgeText: 'text-white'
        };
      default:
        return {
          bg: 'bg-base-100',
          headerBg: 'bg-base-200',
          border: 'border-base-300',
          badgeBg: 'bg-base-300',
          badgeText: 'text-base-content'
        };
    }
  }

  const messageColors = $derived(getMessageColors(message.type));
</script>

<style>
  /* Light mode: bright yellow background, dark text */
  :global(mark.quote-highlight) {
    animation: highlight-fade-light 2s ease-in-out forwards;
    color: rgb(17 24 39) !important; /* gray-900 - dark text for light mode */
  }

  @keyframes highlight-fade-light {
    0% {
      background-color: rgb(253 224 71); /* yellow-300 - bright */
    }
    70% {
      background-color: rgb(253 224 71);
    }
    100% {
      background-color: transparent;
    }
  }

  /* Dark mode: pale yellow background with white text (using class-based dark mode) */
  :global(.dark mark.quote-highlight) {
    animation: highlight-fade-dark 2s ease-in-out forwards;
    color: rgb(255 255 255) !important; /* white text for dark mode */
  }

  @keyframes highlight-fade-dark {
    0% {
      background-color: rgba(253, 224, 71, 0.35); /* pale yellow, 35% opacity */
    }
    70% {
      background-color: rgba(253, 224, 71, 0.35);
    }
    100% {
      background-color: transparent;
    }
  }
</style>

<!-- Message Card Container -->
<div
  class={`border rounded-lg ${messageColors.bg} ${messageColors.border} ${leftEdgeClassForAssistant(message)} ${message.isShared ? 'border-dashed' : ''} ${!isVisible ? 'invisible pointer-events-none' : ''}`}
  data-message-id={message.id}
  style={message.isShared ? 'margin-bottom: 0.5rem;' : ''}
>
  {@render messageHeader()}
  {#if isOpen}
    {@render messageContent()}
  {/if}
</div>

<!-- Message Header Snippet -->
{#snippet messageHeader()}
  <div class="w-full rounded-t-lg {messageColors.headerBg}"><div
      role="button"
      tabindex="0"
      aria-expanded={isOpen}
      class="w-full p-2 flex items-center justify-between hover:bg-black/5 dark:hover:bg-white/5 rounded-t-lg"
      onclick={() => onToggle(message.id || '')}
      onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onToggle(message.id || ''); } }}
    >
      {@render headerBadges()}
      <div class="flex items-center gap-2">
        {@render expandButton()}
        {@render copyMenu()}
      </div>
    </div></div>
{/snippet}

<!-- Header Badges Snippet -->
{#snippet headerBadges()}
  {@const messageName = getMessageSourceLabel(message)}
  <div class="flex items-center gap-2">
    <span class="badge badge-xs font-mono {messageColors.badgeBg} {messageColors.badgeText}">
      {message.type.toUpperCase()}
    </span>
    <span class="badge badge-outline badge-xs">
      Message {messageIndex + 1}
    </span>
    {#if messageName}
      <span class="badge {hashStringToColor(messageName)} badge-sm">{messageName.toUpperCase()}</span>
    {/if}
    {#if message.type === 'tool'}
      {#if message.function}
        <span class="text-xs font-mono">{message.function}</span>
      {/if}
      {#if message.error}
        <span class="badge badge-error badge-xs">ERROR</span>
      {/if}
    {/if}
    <!-- {#if message.isShared}
      <span class="badge badge-neutral badge-xs">SHARED</span>
    {/if} -->
  </div>
{/snippet}

<!-- Expand Button Snippet -->
{#snippet expandButton()}
  <svg 
    class="w-4 h-4 transition-transform {isOpen ? '' : 'rotate-180'}"
    fill="none" 
    stroke="currentColor" 
    viewBox="0 0 24 24"
  >
    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
  </svg>
{/snippet}

<!-- Message Menu Snippet -->
{#snippet copyMenu()}
  <div class="dropdown dropdown-end ml-2" tabindex="0" onkeydown={(e)=>{ if(e.key==='Escape'){ (e.currentTarget as HTMLElement).blur?.(); } }} onclick={(e) => e.stopPropagation()} role="menu" aria-label="Message actions">
    <button type="button" class="btn btn-ghost btn-xs btn-square" aria-label="Message menu">
      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-4 h-4">
        <path stroke-linecap="round" stroke-linejoin="round" d="M12 6.75a.75.75 0 1 1 0-1.5.75.75 0 0 1 0 1.5ZM12 12.75a.75.75 0 1 1 0-1.5.75.75 0 0 1 0 1.5ZM12 18.75a.75.75 0 1 1 0-1.5.75.75 0 0 1 0 1.5Z" />
      </svg>
    </button>
    <ul class="dropdown-content menu bg-base-100 rounded-box z-[1] w-64 p-2 shadow-lg border border-base-300">
      <li>
        <button onclick={(e) => { e.stopPropagation(); toggleRawJsonView(); }} class="text-left">
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-4 h-4">
            <path stroke-linecap="round" stroke-linejoin="round" d="M17.25 6.75 22.5 12l-5.25 5.25m-10.5 0L1.5 12l5.25-5.25m7.5-3-4.5 16.5" />
          </svg>
          {showRawJson ? 'Show rendered view' : 'Show raw JSON'}
        </button>
      </li>
      <div class="divider my-1"></div>
      <li>
        <button onclick={() => onCopy({ type: 'history', columnIndex, messageIndex })} class="text-left">
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-4 h-4">
            <path stroke-linecap="round" stroke-linejoin="round" d="M15.75 17.25v3.375c0 .621-.504 1.125-1.125 1.125h-9.75A1.125 1.125 0 0 1 3.75 20.625V9.375c0-.621.504-1.125 1.125-1.125H7.5a9.75 9.75 0 0 1 1.5.124m5.25 10.501v-3.375c0-.621.504-1.125 1.125-1.125h3.375c.621 0 1.125.504 1.125 1.125v3.375c0 .621-.504 1.125-1.125 1.125h-3.375A1.125 1.125 0 0 1 15.75 17.25Z" />
          </svg>
          Copy history up to here ({messageIndex + 1} messages)
        </button>
      </li>
      <li>
        <button onclick={() => onCopy({ type: 'events', columnIndex, messageIndex })} class="text-left">
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-4 h-4">
            <path stroke-linecap="round" stroke-linejoin="round" d="M3.75 12h16.5m-16.5 3.75h16.5M3.75 19.5h16.5M5.625 4.5h12.75a1.875 1.875 0 0 1 0 3.75H5.625a1.875 1.875 0 0 1 0-3.75Z" />
          </svg>
          Copy events up to here
        </button>
      </li>
      <li>
        <button onclick={() => onCopy({ type: 'single', message })} class="text-left">
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-4 h-4">
            <path stroke-linecap="round" stroke-linejoin="round" d="M8.25 7.5V6.108c0-1.135.845-2.098 1.976-2.192.373-.03.748-.057 1.124-.08M15.75 18H18a2.25 2.25 0 0 0 2.25-2.25V6.108c0-1.135-.845-2.098-1.976-2.192a48.424 48.424 0 0 0-1.123-.08M15.75 18.75v-1.875a3.375 3.375 0 0 0-3.375-3.375h-1.5a1.125 1.125 0 0 1-1.125-1.125v-1.5A3.375 3.375 0 0 0 6.375 7.5H5.25m11.9-3.664A2.251 2.251 0 0 0 15 2.25h-1.5a2.251 2.251 0 0 0-2.15 1.586m5.8 0c.065.21.1.433.1.664v.75h-6V4.5c0-.231.035-.454.1-.664M6.75 7.5H4.875c-.621 0-1.125.504-1.125 1.125v12c0 .621.504 1.125 1.125 1.125h9.75c.621 0 1.125-.504 1.125-1.125V16.5a9 9 0 0 0-9-9Z" />
          </svg>
          Copy this message only
        </button>
      </li>
      <li>
        <button onclick={() => onCopy({ type: 'tools', columnIndex, messageIndex })} class="text-left">
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-4 h-4">
            <path stroke-linecap="round" stroke-linejoin="round" d="M11.42 15.17L17.25 21A2.652 2.652 0 0 0 21 17.25l-5.877-5.877M11.42 15.17l2.496-3.03c.317-.384.74-.626 1.208-.766M11.42 15.17l-4.655 5.653a2.548 2.548 0 1 1-3.586-3.586l6.837-5.63m5.108-.233c.55-.164 1.163-.188 1.743-.14a4.5 4.5 0 0 0 4.486-6.336l-3.276 3.277a3.004 3.004 0 0 1-2.25-2.25l3.276-3.276a4.5 4.5 0 0 0-6.336 4.486c.091 1.076-.071 2.264-.904 2.95l-.102.085m-1.745 1.437L5.909 7.5H4.5L2.25 3.75l1.5-1.5L7.5 4.5v1.409l4.26 4.26m-1.745 1.437l1.745-1.437m6.615 8.206L15.75 15.75M4.867 19.125h.008v.008h-.008v-.008z" />
          </svg>
          Copy tools created up to here
        </button>
      </li>
    </ul>
  </div>
{/snippet}

<!-- Message Content Snippet -->
{#snippet messageContent()}
  <div class="px-3 pb-3 border-t border-base-300 dark:border-white/10">
    {#if showRawJson}
      <div class="mt-2">
        <JsonViewer value={message} theme={$themeString} />
      </div>
    {:else}
      <div class="text-sm leading-relaxed mt-2 message-content text-gray-900 dark:text-gray-100 overflow-x-auto">
        {#if message.type === 'system'}
          {@render systemMessage(message)}
        {:else if message.type === 'user'}
          {@render userMessage(message)}
        {:else if message.type === 'assistant'}
          {@render assistantMessage(message)}
        {:else if message.type === 'tool'}
          {@render toolMessage(message)}
        {:else if message.type === 'api_failure'}
          {@render apiFailureMessage(message)}
        {:else if message.type === 'info'}
          {@render infoMessage(message)}
        {:else}
          <div class="text-red-500">Unknown message type: {JSON.stringify(message)}</div>
        {/if}
      </div>
    {/if}
  </div>
{/snippet}

<!-- System Message Snippet -->
{#snippet systemMessage(message: SystemMessage)}
  {@const textContent = extractTextFromContent(message.content)}
  {#if textContent !== null}
    <span class="text-gray-900 dark:text-gray-100 whitespace-pre-wrap">{@html highlightPlainText(textContent)}</span>
  {:else}
    <JsonViewer value={message.content} theme={$themeString} inlineShortContainers={80} />
  {/if}
{/snippet}

<!-- User Message Snippet -->
{#snippet userMessage(message: UserMessage)}
  {@const textContent = extractTextFromContent(message.content)}
  {#if textContent !== null}
    <span class="text-gray-900 dark:text-gray-100 whitespace-pre-wrap">{@html highlightPlainText(textContent)}</span>
  {:else}
    <JsonViewer value={message.content} theme={$themeString} inlineShortContainers={80} />
  {/if}
{/snippet}

<!-- Assistant Message Snippet -->
{#snippet assistantMessage(message: AssistantMessage)}
  {@const textContent = extractTextFromContent(message.content)}
  {#if message.reasoning}
    <div class="mb-3 p-3 rounded-lg bg-purple-100/50 dark:bg-purple-900/20 border border-purple-200/50 dark:border-purple-700/30">
      <div class="flex items-center gap-2 mb-2">
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-4 h-4 text-purple-600 dark:text-purple-400">
          <path stroke-linecap="round" stroke-linejoin="round" d="M12 18v-5.25m0 0a6.01 6.01 0 0 0 1.5-.189m-1.5.189a6.01 6.01 0 0 1-1.5-.189m3.75 7.478a12.06 12.06 0 0 1-4.5 0m3.75 2.383a14.406 14.406 0 0 1-3 0M14.25 18v-.192c0-.983.658-1.823 1.508-2.316a7.5 7.5 0 1 0-7.517 0c.85.493 1.509 1.333 1.509 2.316V18" />
        </svg>
        <span class="text-xs font-semibold uppercase tracking-wide text-purple-700 dark:text-purple-300">Reasoning</span>
      </div>
      <div class="text-sm text-purple-900 dark:text-purple-100 whitespace-pre-wrap leading-relaxed italic">
        {@html highlightPlainText(message.reasoning.trim())}
      </div>
    </div>
  {/if}
  {#if textContent !== null}
    <span class="text-gray-900 dark:text-gray-100 whitespace-pre-wrap">{@html highlightPlainText(textContent)}</span>
  {:else}
    <JsonViewer value={message.content} theme={$themeString} inlineShortContainers={80} />
  {/if}
  {#if message.tool_calls && message.tool_calls.length > 0}
    {@render toolCalls(message.tool_calls)}
  {/if}
{/snippet}

<!-- Tool Message Snippet -->
{#snippet toolMessage(message: ToolMessage)}
  <div class="space-y-2">
    {#if message.error}
      <div class="whitespace-pre-wrap leading-snug text-[0.9em] text-red-600 dark:text-red-400 overflow-x-auto">
        {@html highlightPlainText(message.error.message)}
      </div>
    {/if}
    {#if message.content}
      <div class="whitespace-pre-wrap font-mono text-[0.9em] leading-snug text-sm text-gray-900 dark:text-gray-100 overflow-x-auto">{@html highlightPlainText(message.content)}</div>
    {/if}
  </div>
{/snippet}

<!-- API Failure Message Snippet -->
{#snippet apiFailureMessage(message: APIFailureMessage)}
  <div class="space-y-2">
    <div class="flex items-center gap-2 mb-2">
      <span class="badge badge-error badge-sm">API FAILURE</span>
      <span class="text-xs text-red-600 dark:text-red-400">{message.error_category}</span>
      {#if message.recoverable}
        <span class="text-xs text-orange-600 dark:text-orange-400">(recoverable)</span>
      {/if}
    </div>
    <div class="font-mono text-sm bg-red-50 dark:bg-red-900/20 p-2 rounded border border-red-200 dark:border-red-700/50 text-gray-900 dark:text-red-100">
      {@html highlightPlainText(message.error_message)}
    </div>
  </div>
{/snippet}

<!-- Info Message Snippet -->
{#snippet infoMessage(message: InfoMessage)}
  <div class="space-y-2">
    <div class="flex items-center gap-2 mb-2">
      <span class="badge badge-info badge-sm">INFO</span>
      {#if message.timestamp}
        <span class="text-xs text-gray-500 dark:text-gray-400">{new Date(message.timestamp).toLocaleString()}</span>
      {/if}
      {#if message.id}
        <span class="text-xs text-gray-500 dark:text-gray-400">ID: {message.id}</span>
      {/if}
    </div>
    <div class="font-mono text-sm bg-blue-50 dark:bg-blue-900/20 p-2 rounded border border-blue-200 dark:border-blue-700/50 text-gray-900 dark:text-blue-100">
      {#if typeof message.info === 'string'}
        {@html highlightPlainText(message.info)}
      {:else}
        <JsonViewer value={message.info} theme={$themeString} inlineShortContainers={80} />
      {/if}
    </div>
  </div>
{/snippet}

<!-- Tool Calls Snippet -->
{#snippet toolCalls(toolCallsList: ToolCall[])}
  <div class="mt-3">
    {#each toolCallsList as toolCall, i}
      {@const tcId = toolCall.id || String(i)}
      {#if i > 0}
        <div class="my-2 border-t border-dashed border-base-300 dark:border-white/20"></div>
      {/if}
      <div class="grid grid-cols-[16px_1fr] gap-x-2 items-start">
        <div class="flex items-center justify-center">
          <button
            type="button"
            class="p-0.5 rounded hover:bg-black/5 dark:hover:bg-white/5"
            aria-label="Toggle tool call"
            aria-expanded={!isToolCallCollapsed(tcId)}
            aria-controls={`toolcall-${tcId}`}
            onclick={(e) => { e.stopPropagation(); toggleToolCallCollapsed(tcId); }}
            onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); e.stopPropagation(); toggleToolCallCollapsed(tcId); } }}
          >
            <svg class="w-3 h-3 transition-transform {isToolCallCollapsed(tcId) ? '-rotate-90' : 'rotate-0'}" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
            </svg>
          </button>
        </div>
        <div class="text-xs mb-1 flex items-center overflow-x-auto">
          <span class="badge badge-info badge-xs mr-1 flex-shrink-0">TOOL</span>
          <span class="font-mono flex-shrink-0">{toolCall.function}</span>
          <span class="text-gray-500 dark:text-gray-400 ml-2 flex-shrink-0">ID: {toolCall.id}</span>
        </div>
        {#if !isToolCallCollapsed(tcId)}
          <div class="col-start-1 self-stretch justify-self-center">
            <div class="h-full w-px bg-base-300 dark:bg-white/20 mt-[2px] mx-auto"></div>
          </div>
          {#if toolCall.view}
            {#if toolCall.view.format === 'markdown'}
              <div id={`toolcall-${tcId}`} class="prose max-w-none text-sm [font-size-adjust:from-font] prose-p:my-0 prose-ul:my-0 prose-ol:my-0 prose-headings:my-1 prose-blockquote:my-0 prose-pre:my-0 prose-pre:p-0 prose-code:my-0 prose-pre:bg-transparent prose-code:bg-transparent prose-pre:text-inherit prose-code:text-inherit prose-pre:leading-snug prose-pre:whitespace-pre-wrap prose-code:whitespace-pre-wrap prose-pre:break-words prose-code:break-words prose-table:text-xs overflow-x-auto">{@html renderMarkdown(toolCall.view.content)}</div>
            {:else}
              <div id={`toolcall-${tcId}`} class="whitespace-pre-wrap leading-snug font-mono text-[0.9em] overflow-x-auto">{toolCall.view.content}</div>
            {/if}
          {:else if toolCall.arguments && Object.keys(toolCall.arguments || {}).length > 0}
            <div id={`toolcall-${tcId}`} class="whitespace-pre-wrap leading-snug font-mono text-[0.9em] overflow-x-auto">{toYaml(toolCall.arguments)}</div>
          {/if}
        {/if}
      </div>
    {/each}
  </div>
{/snippet}