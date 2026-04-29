<script lang="ts">
  import type { Message } from '$lib/shared/types';
  import type { ConversationColumn } from '$lib/shared/transcript-parser';
  import { createTranscriptLoader } from '$lib/client/utils/transcript.svelte';
  import { parseTranscriptEvents, extractAvailableViews } from '$lib/shared/transcript-parser';
  import { handleCopyAction, type CopyAction } from '$lib/client/utils/copy-utils';
  import MessageCard from './MessageCard.svelte';
  import ScoreTooltip from '$lib/client/components/common/ScoreTooltip.svelte';
  import { JsonViewer } from '@kaifronsdal/svelte-json-viewer';

  interface Props {
    filePath: string;
  }

  let { filePath }: Props = $props();

  // Load transcript using our new loader
  const loader = createTranscriptLoader(filePath);

  // Extract suite name, config name, and transcript ID from file path
  let suiteAndConfig = $derived.by(() => {
    const pathParts = filePath.split('/');
    if (pathParts.length >= 3) {
      const suite = pathParts[0];
      const config = pathParts[1];
      const filename = pathParts[pathParts.length - 1];
      const transcriptId = filename.replace(/^transcript_/, '').replace(/\.json$/, '');
      return { suite, config, transcriptId };
    }
    return { suite: '', config: '', transcriptId: loader.transcript?.id || '' };
  });

  // View settings state
  let selectedView = $state('combined');
  let showApiFailures = $state(false);
  let showSharedHistory = $state(true);
  let showSystemPrompt = $state(false);

  // Message state management
  let openMessages: Record<string, boolean> = $state({});

  // Message refs for scrolling
  let messageRefs: Map<string, HTMLElement> = new Map();

  // Store the current highlighted quote text
  let highlightedQuoteText = $state<string | null>(null);

  // Parse conversation columns from loaded transcript
  let conversationColumns = $derived.by(() => {
    if (!loader.transcript?.transcript?.events || selectedView === 'raw') {
      return [];
    }
    
    return parseTranscriptEvents(loader.transcript.transcript.events, selectedView, showApiFailures);
  });

  // Extract available views from loaded transcript
  let availableViews = $derived.by(() => {
    if (!loader.transcript?.transcript?.events) {
      return ['combined'];
    }
    
    return extractAvailableViews(loader.transcript.transcript.events);
  });

  // Auto-load transcript on mount
  $effect(() => {
    loader.loadTranscript();
  });

  // Message toggle functionality
  function toggleMessage(messageId: string) {
    // Since default is true, toggle means: if not false, set to false; if false, set to true
    const newState = openMessages[messageId] === false ? true : false;
    openMessages[messageId] = newState;
  }

  function isMessageOpen(messageId: string): boolean {
    // Default to true (open) if not explicitly set to false
    return openMessages[messageId] !== false;
  }

  // Copy action handler using the utilities
  async function onCopyAction(action: CopyAction) {
    const result = await handleCopyAction(
      action, 
      conversationColumns, 
      loader.transcript?.transcript.events
    );
    
    // TODO: Add toast notification system
    console.log(result.message);
    
    if (result.isError) {
      console.error('Copy failed:', result.message);
    }
  }

  // Helper to check if a shared message should be visible
  function shouldShowSharedMessage(message: Message, messageIndex: number, columnMessages: Message[]): boolean {
    if (!message.isShared) return true;
    return showSharedHistory;
  }

  // Utility function to convert string to title case
  function toTitleCase(str: string): string {
    return str.replace(/\w\S*/g, (txt) =>
      txt.charAt(0).toUpperCase() + txt.slice(1).toLowerCase()
    );
  }

  // Scroll to a specific message by ID and highlight specific quoted text
  function scrollToMessage(messageId: string, quotedText?: string) {
    const element = messageRefs.get(messageId);
    if (element) {
      // Expand the message if it's collapsed
      if (openMessages[messageId] === false) {
        openMessages[messageId] = true;
      }

      // Store the quoted text to highlight
      if (quotedText) {
        highlightedQuoteText = quotedText;
      }

      // Wait a tick for the message to expand
      setTimeout(() => {
        // Scroll to the message with smooth behavior
        element.scrollIntoView({ behavior: 'smooth', block: 'center' });

        // Add a highlight effect to the message card
        element.classList.add('highlight-flash');
        setTimeout(() => {
          element.classList.remove('highlight-flash');
          // Clear the highlighted quote after animation
          highlightedQuoteText = null;
        }, 2000);
      }, 0);
    }
  }

  // Build a map of quoted text to message IDs from highlights
  // Structure: Map<fullQuotedText, { messageId, quotedText }>
  let quoteToMessageMap = $derived.by(() => {
    const map = new Map<string, { messageId: string; quotedText: string }>();

    // Access highlights from the transcript metadata
    const highlights = loader.transcript?.transcript?.metadata?.judge_output?.highlights;

    if (!highlights) return map;

    for (const highlight of highlights) {
      for (const part of highlight.parts) {
        map.set(part.quoted_text, {
          messageId: part.message_id,
          quotedText: part.quoted_text
        });
      }
    }

    return map;
  });

  // Render justification text with clickable quote links
  function renderJustificationWithLinks(text: string, quotes: string[]): string {
    let result = text;
    const replacements: Array<{ start: number; end: number; html: string; messageId: string }> = [];
    let refNumber = 1;

    // Extract all quoted strings from the justification text using regex
    // Matches text between double quotes, handling escaped quotes
    const quoteRegex = /"([^"]+)"/g;
    let match;

    while ((match = quoteRegex.exec(text)) !== null) {
      const quotedText = match[1]; // Text without the surrounding quotes
      const fullMatch = match[0];  // Text with the surrounding quotes
      const startPos = match.index;
      const endPos = startPos + fullMatch.length;

      // Find if this quoted text matches any part of our highlight quotes
      for (const [highlightQuote, data] of quoteToMessageMap.entries()) {
        if (highlightQuote.includes(quotedText) || quotedText.length > 20 && highlightQuote.toLowerCase().includes(quotedText.toLowerCase())) {
          // Check if this position is already part of a replacement
          const isOverlapping = replacements.some(
            (r) => (startPos >= r.start && startPos < r.end) ||
                   (endPos > r.start && endPos <= r.end)
          );

          if (!isOverlapping) {
            // Escape quotes in the quoted text for data attributes
            const escapedQuotedText = data.quotedText.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
            // Use data attributes instead of inline onclick
            const refLink = `<button class="quote-ref-btn text-primary hover:text-primary-focus font-semibold cursor-pointer ml-0.5 underline" data-message-id="${data.messageId}" data-quoted-text="${escapedQuotedText}" title="Jump to message">[${refNumber}]</button>`;
            const newHtml = fullMatch + refLink;
            replacements.push({
              start: startPos,
              end: endPos,
              html: newHtml,
              messageId: data.messageId
            });
            refNumber++;
            break; // Only match once per quoted string
          }
        }
      }
    }

    // Sort replacements by start position in reverse order
    replacements.sort((a, b) => b.start - a.start);

    // Apply replacements
    for (const replacement of replacements) {
      result = result.slice(0, replacement.start) + replacement.html + result.slice(replacement.end);
    }

    return result;
  }

  // Listen for clicks on quote reference buttons using event delegation
  $effect(() => {
    function handleQuoteClick(event: MouseEvent) {
      const target = event.target as HTMLElement;

      // Check if the clicked element is a quote reference button
      if (target.classList.contains('quote-ref-btn')) {
        const messageId = target.getAttribute('data-message-id');
        const quotedText = target.getAttribute('data-quoted-text');

        if (messageId) {
          // Decode HTML entities
          const decodedQuotedText = quotedText ? quotedText.replace(/&quot;/g, '"').replace(/&#39;/g, "'") : undefined;
          scrollToMessage(messageId, decodedQuotedText);
        }
      }
    }

    document.addEventListener('click', handleQuoteClick);
    return () => {
      document.removeEventListener('click', handleQuoteClick);
    };
  });

  // Svelte action to register message refs
  function registerMessageRef(node: HTMLElement, messageId: string) {
    messageRefs.set(messageId, node);
    return {
      destroy() {
        messageRefs.delete(messageId);
      }
    };
  }



  // Horizontal overflow detection for smart centering
  let scrollContainer = $state<HTMLDivElement | null>(null);
  let hasHorizontalOverflow = $state(false);

  function updateOverflowState() {
    if (!scrollContainer) {
      hasHorizontalOverflow = false;
      return;
    }
    hasHorizontalOverflow = scrollContainer.scrollWidth > scrollContainer.clientWidth + 4;
  }

  // Keep overflow state in sync
  $effect(() => {
    updateOverflowState();
  });

  $effect(() => {
    function onResize() {
      updateOverflowState();
    }
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  });

  // Ensure overflow state updates after DOM changes when container or columns change
  $effect(() => {
    // Capture dependencies so this runs when they change
    const container = scrollContainer;
    const numCols = conversationColumns.length;
    // Schedule after layout
    requestAnimationFrame(() => {
      updateOverflowState();
    });
  });

</script>

<style>
  :global(.highlight-flash) {
    animation: highlight-pulse 2s ease-in-out;
  }

  @keyframes highlight-pulse {
    0%, 100% {
      box-shadow: 0 0 0 0 rgba(59, 130, 246, 0);
    }
    50% {
      box-shadow: 0 0 20px 5px rgba(59, 130, 246, 0.5);
    }
  }
</style>

<div class="w-full">
  <!-- Loading State -->
  {#if loader.transcriptLoading}
    <div class="flex items-center justify-center p-8">
      <span class="loading loading-spinner loading-lg"></span>
      <span class="ml-4">Loading transcript...</span>
    </div>
  {/if}

  <!-- Error State -->
  {#if loader.transcriptError}
    <div class="alert alert-error max-w-6xl mx-auto m-4">
      <svg xmlns="http://www.w3.org/2000/svg" class="stroke-current shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
      <span>{loader.transcriptError}</span>
    </div>
  {/if}

  <!-- Loaded Content -->
  {#if loader.transcript}
    {@render transcriptContent()}
  {/if}
</div>

<!-- Transcript Content Snippet -->
{#snippet transcriptContent()}
  <!-- Metadata Section - Constrained width -->
  <div class="p-4 space-y-6">
    <!-- Header with metadata -->
    {@render transcriptHeader()}
    
    <!-- Conversation Section with controls -->
    {@render viewControls()}
  </div>

  <!-- Content Section - Full width -->
  {#if selectedView === 'raw'}
    {@render rawJsonView()}
  {:else}
    {@render conversationView()}
  {/if}
{/snippet}

<!-- Transcript Header Snippet -->
{#snippet transcriptHeader()}
  <div class="card bg-base-100 shadow-sm max-w-6xl mx-auto">
    <div class="card-body">
      <div class="flex justify-between items-start mb-4">
        <div>
          <h1 class="text-2xl font-bold mb-2">
            {suiteAndConfig.suite} - {suiteAndConfig.config}
          </h1>
          <p class="text-base-content/70">Transcript {suiteAndConfig.transcriptId}</p>
        </div>
      </div>

      <!-- Score Grid -->
      <div class="mb-6">
        <h3 class="text-lg font-semibold mb-3">Scores</h3>
        <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-2">
          {#each Object.entries(loader.transcript?.scores || {}) as [key, value]}
            <ScoreTooltip 
              score={value} 
              scoreName={key} 
              description={loader.transcript?.scoreDescriptions?.[key]}
            />
          {/each}
        </div>
      </div>


      <!-- Judge Summary -->
      <div class="mb-4">
        <h3 class="text-lg font-semibold mb-2">Judge Summary</h3>
        <p class="text-sm leading-relaxed">{loader.transcript?.judgeSummary}</p>
      </div>

      <!-- Judge Justification -->
      <div class="mb-4">
        <h3 class="text-lg font-semibold mb-2">Judge Justification</h3>
        {@render justificationContent()}
        {#if quoteToMessageMap.size > 0}
          <div class="flex items-start gap-2 mt-3 text-xs text-base-content/70">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" class="stroke-current shrink-0 w-4 h-4 mt-0.5">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
            </svg>
            <span>
              Numbered references [1], [2], etc. are clickable and will jump to the corresponding message in the transcript.
              Note: References only work if the message is visible in the current view (Evaluator/Target/Combined).
            </span>
          </div>
        {/if}
      </div>

      <!-- System Prompt (Collapsible) -->
      {#if loader.transcript?.systemPrompt}
        <div class="collapse collapse-arrow bg-base-200">
          <input type="checkbox" bind:checked={showSystemPrompt} />
          <div class="collapse-title text-lg font-semibold">
            Agent System Prompt
          </div>
          <div class="collapse-content">
            <div class="bg-base-300 p-4 rounded-lg text-sm font-mono whitespace-pre-wrap">
              {loader.transcript.systemPrompt}
            </div>
          </div>
        </div>
      {/if}

      {@render roundRefinementSection()}
    </div>
  </div>
{/snippet}

<!-- Round Refinement Section Snippet -->
{#snippet roundRefinementSection()}
  {@const rawMeta = (loader.transcript?.transcript?.metadata as any)}
  {@const refinedStrategy = rawMeta?.refined_strategy ?? null}
  {@const refinement = rawMeta?.refinement ?? null}
  {#if refinedStrategy || refinement}
    <div class="space-y-3 pt-2">
      <!-- Strategy injected into this round (from prior round's refinement) -->
      {#if refinedStrategy}
        <div class="p-3 rounded-lg bg-indigo-50/60 dark:bg-indigo-950/25 border border-dashed border-indigo-300 dark:border-indigo-700/50">
          <div class="flex items-center gap-1.5 mb-1.5">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-3.5 h-3.5 text-indigo-600 dark:text-indigo-400 flex-shrink-0">
              <path stroke-linecap="round" stroke-linejoin="round" d="M9.813 15.904 9 18.75l-.813-2.846a4.5 4.5 0 0 0-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 0 0 3.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 0 0 3.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 0 0-3.09 3.09Z" />
            </svg>
            <span class="text-xs font-semibold uppercase tracking-wide text-indigo-700 dark:text-indigo-400">Strategy from prior round</span>
          </div>
          <div class="text-sm text-indigo-900 dark:text-indigo-100 whitespace-pre-wrap leading-relaxed italic">
            {refinedStrategy}
          </div>
        </div>
      {/if}

      <!-- Refinement output derived from this round (feeds into the next round) -->
      {#if refinement}
        <div class="p-3 rounded-lg bg-teal-50/60 dark:bg-teal-950/25 border border-dashed border-teal-300 dark:border-teal-700/50">
          <div class="flex items-center gap-1.5 mb-2">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-3.5 h-3.5 text-teal-600 dark:text-teal-400 flex-shrink-0">
              <path stroke-linecap="round" stroke-linejoin="round" d="M4.26 10.147a60.438 60.438 0 0 0-.491 6.347A48.62 48.62 0 0 1 12 20.904a48.62 48.62 0 0 1 8.232-4.41 60.46 60.46 0 0 0-.491-6.347m-15.482 0a50.636 50.636 0 0 0-2.658-.813A59.906 59.906 0 0 1 12 3.493a59.903 59.903 0 0 1 10.399 5.84c-.896.248-1.783.52-2.658.814m-15.482 0A50.717 50.717 0 0 1 12 13.489a50.702 50.702 0 0 1 3.741-1.342M6.75 15a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5Zm0 0v-3.675A55.378 55.378 0 0 1 12 8.443m-7.007 11.55A5.981 5.981 0 0 0 6.75 15.75v-1.5" />
            </svg>
            <span class="text-xs font-semibold uppercase tracking-wide text-teal-700 dark:text-teal-400">Refinement output <span class="font-normal normal-case">(feeds into next round)</span></span>
          </div>
          {#if refinement.observations}
            <div class="mb-2">
              <span class="text-xs font-semibold text-teal-700 dark:text-teal-400">Observations</span>
              <div class="text-sm text-teal-900 dark:text-teal-100 whitespace-pre-wrap leading-relaxed mt-0.5">
                {refinement.observations}
              </div>
            </div>
          {/if}
          {#if refinement.updated_strategy}
            <div>
              <span class="text-xs font-semibold text-teal-700 dark:text-teal-400">Updated strategy</span>
              <div class="text-sm text-teal-900 dark:text-teal-100 whitespace-pre-wrap leading-relaxed italic mt-0.5">
                {refinement.updated_strategy}
              </div>
            </div>
          {/if}
        </div>
      {/if}
    </div>
  {/if}
{/snippet}

<!-- View Controls Snippet -->
{#snippet viewControls()}
  <div class="card bg-base-100 shadow-sm max-w-6xl mx-auto">
    <div class="card-body">
      <h2 class="text-xl font-bold mb-4">Conversation</h2>
      

      
      <!-- Tab Navigation -->
      <div class="tabs tabs-boxed justify-center mb-4">
        {#each availableViews as view}
          <button
            class="tab {selectedView === view ? 'tab-active' : ''}"
            onclick={() => selectedView = view}
          >
            {toTitleCase(view)}
          </button>
        {/each}
        <button
          class="tab {selectedView === 'raw' ? 'tab-active' : ''}"
          onclick={() => selectedView = 'raw'}
        >
          Raw JSON
        </button>
      </div>

      <!-- Additional Controls (only for conversation views) -->
      {#if selectedView !== 'raw'}
        <div class="flex justify-center mb-4 gap-6">
          <div class="form-control">
            <label class="label cursor-pointer">
              <span class="label-text mr-2">Show API Failures</span>
              <input type="checkbox" class="toggle toggle-error" bind:checked={showApiFailures} />
            </label>
          </div>
          {#if conversationColumns.length > 1}
            <div class="form-control">
              <label class="label cursor-pointer">
                <span class="label-text mr-2">Show Shared History</span>
                <input type="checkbox" class="toggle toggle-primary" bind:checked={showSharedHistory} />
              </label>
            </div>
          {/if}
        </div>
      {/if}
      
      <!-- Content will be rendered outside this card -->
    </div>
  </div>
{/snippet}

<!-- Raw JSON View Snippet -->
{#snippet rawJsonView()}
  <div class="w-full p-4">
    <div class="bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4 max-w-6xl mx-auto">
      <div class="flex items-center justify-between mb-4">
        <h3 class="font-semibold text-lg">Raw Event Data</h3>
        <span class="text-sm text-gray-600 dark:text-gray-400">
          {loader.transcript?.transcript?.events?.length || 0} events
        </span>
      </div>
      <div class="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded p-4">
        <pre class="text-xs overflow-auto min-h-[70vh] whitespace-pre-wrap font-mono">
          {JSON.stringify(loader.transcript?.transcript, null, 2)}
        </pre>
      </div>
    </div>
  </div>
{/snippet}

<!-- Conversation View Snippet -->
{#snippet conversationView()}
  {#if conversationColumns.length === 0}
    <div class="p-4">
      <div class="card bg-base-200 p-8 text-center max-w-6xl mx-auto">
        <p class="text-gray-500">No conversation data available for the selected view.</p>
      </div>
    </div>
  {:else}
    <div class="relative w-full">

      <div
        class="overflow-x-auto overscroll-x-contain scroll-smooth"
        bind:this={scrollContainer}
      >
        <div class="flex gap-6 snap-x snap-mandatory px-6 max-w-6xl mx-auto" style={hasHorizontalOverflow ? '' : 'justify-content: center;'}>
          {#each conversationColumns as column, columnIndex}
            <div class="snap-start flex-1 min-w-[500px] max-w-full">
              {@render conversationColumn(column, columnIndex)}
            </div>
          {/each}
        </div>
      </div>
    </div>
  {/if}
{/snippet}

<!-- Justification Content Snippet -->
{#snippet justificationContent()}
  {@const justification = loader.transcript?.justification || ''}
  {@const quotes = Array.from(quoteToMessageMap.keys()).sort((a, b) => b.length - a.length)}

  <div class="text-sm leading-relaxed">
    {#if quotes.length === 0}
      <p>{justification}</p>
    {:else}
      {@html renderJustificationWithLinks(justification, quotes)}
    {/if}
  </div>
{/snippet}

<!-- Conversation Column Snippet -->
{#snippet conversationColumn(column: ConversationColumn, columnIndex: number)}
  <div class="space-y-4">
    <!-- Column Header -->
    <div class="card bg-base-300 p-3">
      <h3 class="font-semibold text-sm">
        {column.title}
      </h3>
      <p class="text-xs text-gray-500">{column.messages.length} messages</p>
    </div>

    <!-- Messages -->
    <div class="space-y-2">
      {#each column.messages as message, messageIndex}
        {@const isVisible = shouldShowSharedMessage(message, messageIndex, column.messages)}
        {@const messageId = message.id || ''}
        <div
          data-message-id={messageId}
          use:registerMessageRef={messageId}
        >
          <MessageCard
            {message}
            {messageIndex}
            {columnIndex}
            isOpen={isMessageOpen(messageId)}
            {isVisible}
            onToggle={toggleMessage}
            onCopy={onCopyAction}
            highlightText={highlightedQuoteText}
          />
        </div>
      {/each}
    </div>
  </div>
{/snippet}