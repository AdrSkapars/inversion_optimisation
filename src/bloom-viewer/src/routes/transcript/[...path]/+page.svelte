<script lang="ts">
  import { page } from '$app/stores';
  import { parseUrlPath, generateTranscriptUrl } from '$lib/client/utils/file-utils';
  import TranscriptViewer from '$lib/client/components/transcript/TranscriptViewer.svelte';
  
  // Parse the file path from URL parameters
  let filePath = $derived.by(() => {
    const pathSegments = $page.params.path;
    
    if (!pathSegments) {
      return '';
    }
    
    // Handle both string and array path formats
    const segments = Array.isArray(pathSegments) 
      ? pathSegments 
      : pathSegments.split('/').filter(Boolean);
    
    try {
      console.log('Parsing URL segments:', segments);
      const result = parseUrlPath(segments);
      console.log('Parsed file path:', result);
      return result;
    } catch (error) {
      console.error('Failed to parse URL path:', error);
      console.error('Segments were:', segments);
      return '';
    }
  });
  
  // Create breadcrumb segments from file path
  let breadcrumbSegments = $derived.by(() => {
    if (!filePath) return [];

    const pathParts = filePath.split('/');
    const segments = [];

    // Build cumulative paths for each segment, skipping the rootDir (first segment)
    // The first segment is typically the rootDir (e.g., "transcripts"),
    // so we want paths relative to that
    for (let i = 0; i < pathParts.length; i++) {
      const segment = pathParts[i];
      const isFile = i === pathParts.length - 1 && segment.endsWith('.json');

      // For directories, create path relative to rootDir
      let relativePath = pathParts.slice(0, i + 1).join('/');

      segments.push({
        name: segment,
        path: relativePath,
        isFile,
        isClickable: !isFile && i > 0 // Only directories after rootDir are clickable
      });
    }

    return segments;
  });

  // Generate page title: "eval suite - config - transcript id"
  let pageTitle = $derived.by(() => {
    if (!filePath) return 'Bloom Transcript Viewer';

    const pathParts = filePath.split('/');
    // Expected structure: [suite, config, transcript_xxx.json]
    if (pathParts.length >= 3) {
      const suite = pathParts[0];
      const config = pathParts[1];
      const filename = pathParts[pathParts.length - 1];
      const transcriptId = filename.replace(/^transcript_/, '').replace(/\.json$/, '');
      return `${suite} - ${config} - ${transcriptId}`;
    }
    return filePath;
  });

  let errorMessage = $derived(filePath ? '' : 'Invalid transcript path');
</script>

<svelte:head>
  <title>{pageTitle}</title>
</svelte:head>

<div class="min-h-screen bg-base-100">
  <div class="navbar bg-base-200">
    <div class="flex-1">
      <div class="breadcrumbs text-sm">
        <ul>
          <li><a href="/" class="font-mono text-xs">Home</a></li>
          {#each breadcrumbSegments as segment, index}
            <li>
              {#if segment.isFile}
                <!-- Current file - not clickable -->
                <span class="font-mono text-xs font-semibold">
                  {segment.name}
                </span>
              {:else if index === 0}
                <!-- First segment (suite) - link to root with no path -->
                <a
                  href="/"
                  class="font-mono text-xs transition-colors"
                  title="View all suites"
                >
                  {segment.name}
                </a>
              {:else}
                <!-- Directory - link to appropriate path in new viewer -->
                <a
                  href="/?path={encodeURIComponent(breadcrumbSegments.slice(0, index + 1).map(s => s.name).join('/'))}"
                  class="font-mono text-xs transition-colors"
                  title="View directory: {segment.name}"
                >
                  {segment.name}
                </a>
              {/if}
            </li>
          {/each}
        </ul>
      </div>
    </div>
  </div>

  <main class="py-6">
    {#if errorMessage}
      <div class="p-4">
        <div class="alert alert-error max-w-2xl mx-auto">
          <svg xmlns="http://www.w3.org/2000/svg" class="stroke-current shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span>{errorMessage}</span>
          <div>
            <a href="/" class="btn btn-sm">Go Home</a>
          </div>
        </div>
      </div>
    {:else}
      <TranscriptViewer {filePath} />
    {/if}
  </main>
</div>