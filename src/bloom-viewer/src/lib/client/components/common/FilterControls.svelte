<script lang="ts">
  import { filterState } from '$lib/client/stores';
  import { 
    validateFilterExpression, 
    getFilterExamples, 
    getAvailableFields, 
    getCurrentWord, 
    getAutocompleteSuggestions 
  } from '$lib/shared/filter-utils';

  interface Props {
    scoreTypes: string[];
    filteredCount: number;
    totalCount: number;
  }

  let { scoreTypes, filteredCount, totalCount }: Props = $props();

  // Autocomplete state
  let filterInput: HTMLInputElement;
  let showAutocomplete = $state(false);
  let autocompleteIndex = $state(-1);
  let currentWord = $state({ word: '', startPos: 0, endPos: 0 });
  let suggestions = $state<string[]>([]);

  // Get filter examples for help
  let filterExamples = $derived(getFilterExamples());
  
  // Get available fields for autocomplete
  let availableFields = $derived(getAvailableFields(scoreTypes));
  
  // Validate filter expression
  let filterError = $derived.by(() => {
    return validateFilterExpression(filterState.value.filterExpression);
  });

  // Autocomplete functions
  function updateAutocomplete() {
    if (!filterInput) return;
    
    const cursorPosition = filterInput.selectionStart || 0;
    const text = filterState.value.filterExpression || '';
    
    currentWord = getCurrentWord(text, cursorPosition);
    suggestions = getAutocompleteSuggestions(currentWord.word, availableFields);
    
    showAutocomplete = suggestions.length > 0 && currentWord.word.length > 0;
    autocompleteIndex = -1;
  }

  function handleFilterInput() {
    updateAutocomplete();
  }

  function handleFilterKeydown(event: KeyboardEvent) {
    if (!showAutocomplete) return;
    
    if (event.key === 'ArrowDown') {
      event.preventDefault();
      autocompleteIndex = Math.min(autocompleteIndex + 1, suggestions.length - 1);
    } else if (event.key === 'ArrowUp') {
      event.preventDefault();
      autocompleteIndex = Math.max(autocompleteIndex - 1, -1);
    } else if (event.key === 'Enter' && autocompleteIndex >= 0) {
      event.preventDefault();
      selectSuggestion(suggestions[autocompleteIndex]);
    } else if (event.key === 'Escape') {
      showAutocomplete = false;
      autocompleteIndex = -1;
    }
  }

  function selectSuggestion(suggestion: string) {
    const text = filterState.value.filterExpression || '';
    const newText = text.slice(0, currentWord.startPos) + suggestion + text.slice(currentWord.endPos);
    
    filterState.value.filterExpression = newText;
    showAutocomplete = false;
    autocompleteIndex = -1;
    
    // Set cursor position after the inserted suggestion
    setTimeout(() => {
      if (filterInput) {
        const newCursorPos = currentWord.startPos + suggestion.length;
        filterInput.setSelectionRange(newCursorPos, newCursorPos);
        filterInput.focus();
      }
    }, 0);
  }

  function handleClickOutside(event: MouseEvent) {
    const target = event.target as Element;
    if (!target.closest('.autocomplete-container')) {
      showAutocomplete = false;
      autocompleteIndex = -1;
    }
  }
</script>

<div class="card bg-base-100 shadow-sm">
  <div class="card-body">
    <h2 class="text-xl font-semibold mb-4">Filters</h2>
    
    <!-- Search Query -->
    <div class="form-control mb-4">
      <label class="label">
        <span class="label-text">Search in summary</span>
      </label>
      <input 
        type="text" 
        placeholder="Search transcripts by summary text..." 
        class="input input-bordered w-full"
        bind:value={filterState.value.searchQuery}
      />
    </div>

    <!-- Filter Expression -->
    <div class="form-control mb-4 autocomplete-container relative">
      <label class="label">
        <span class="label-text">Advanced Filter Expression</span>
        <div class="dropdown dropdown-end">
          <div tabindex="0" role="button" class="btn btn-ghost btn-xs">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" class="w-4 h-4 stroke-current">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
            </svg>
          </div>
          <div class="dropdown-content z-[1] menu p-2 shadow bg-base-100 rounded-box w-80">
            <div class="text-sm">
              <h3 class="font-semibold mb-2">Filter Examples:</h3>
              <ul class="space-y-1">
                {#each filterExamples as example}
                  <li>
                    <code class="text-xs bg-base-200 px-1 py-0.5 rounded">{example.expression}</code>
                    <br>
                    <span class="text-xs text-base-content/70">{example.description}</span>
                  </li>
                {/each}
              </ul>
            </div>
          </div>
        </div>
      </label>
      
      <input 
        bind:this={filterInput}
        type="text" 
        placeholder="e.g., concerningScore > 5 && model = 'claude-3.5'"
        class="input input-bordered w-full {filterError ? 'input-error' : ''}"
        bind:value={filterState.value.filterExpression}
        oninput={handleFilterInput}
        onkeydown={handleFilterKeydown}
        autocomplete="off"
      />
      
      {#if filterError}
        <label class="label">
          <span class="label-text-alt text-error">{filterError}</span>
        </label>
      {/if}

      <!-- Autocomplete dropdown -->
      {#if showAutocomplete}
        <div class="absolute top-full left-0 right-0 bg-base-100 border border-base-300 rounded-box shadow-lg z-10 max-h-48 overflow-y-auto">
          {#each suggestions as suggestion, index}
            <button
              type="button"
              class="w-full text-left px-3 py-2 hover:bg-base-200 {index === autocompleteIndex ? 'bg-base-200' : ''}"
              onclick={() => selectSuggestion(suggestion)}
            >
              <code class="text-sm">{suggestion}</code>
            </button>
          {/each}
        </div>
      {/if}
    </div>

    <!-- Results Counter -->
    <div class="flex justify-between items-center text-sm text-base-content/70">
      <span>
        Showing {filteredCount.toLocaleString()} of {totalCount.toLocaleString()} transcripts
      </span>
      {#if filteredCount !== totalCount}
        <button 
          class="btn btn-ghost btn-xs"
          onclick={() => {
            filterState.value.searchQuery = '';
            filterState.value.filterExpression = '';
          }}
        >
          Clear filters
        </button>
      {/if}
    </div>
  </div>
</div>

<svelte:window onclick={handleClickOutside} />