<script lang="ts">
	import { onMount } from 'svelte';
	import { page } from '$app/stores';
	import TranscriptTable from '$lib/client/components/TranscriptTable.svelte';
	import FilterControls from '$lib/client/components/common/FilterControls.svelte';
	import ViewModeToggle from '$lib/client/components/common/ViewModeToggle.svelte';
	import ErrorDisplay from '$lib/client/components/ErrorDisplay.svelte';
	import MetajudgeReport from '$lib/client/components/common/MetajudgeReport.svelte';
	import { filterState, viewSettings, initializeStores } from '$lib/client/stores';
	import { createFilterFunction } from '$lib/shared/filter-utils';
	import { createTranscriptDataLoader } from '$lib/shared/services/transcript-data.svelte';
	import { extractAllTranscriptsFromTree, filterFolderTree } from '$lib/client/utils/folder-tree';
	import { collectScoreDescriptions } from '$lib/shared/utils/transcript-utils';
	import { getScoreColorContinuous } from '$lib/shared/score-utils';
	// Live update SSE removed for simplified API

	// Create data loader
	const dataLoader = createTranscriptDataLoader();

	// Get judgment and evaluation data from the loaded index
	// This replaces the separate API calls
	function getJudgmentData(configPath: string): any {
		const data = dataLoader.getConfigData(configPath);
		return data?.judgment || null;
	}

	function getEvaluationMetadata(configPath: string): any {
		const data = dataLoader.getConfigData(configPath);
		return data?.evaluation || null;
	}

	// Get current subdirectory path from URL parameter
	let currentPath = $derived($page.url.searchParams.get('path') || '');
	
	// Create breadcrumb segments for current path
	let breadcrumbSegments = $derived.by(() => {
		if (!currentPath) return [];
		
		const pathParts = currentPath.split('/').filter(Boolean);
		const segments = [];
		
		// Build cumulative paths for each segment
		for (let i = 0; i < pathParts.length; i++) {
			const segment = pathParts[i];
			const cumulativePath = pathParts.slice(0, i + 1).join('/');
			
			segments.push({
				name: segment,
				path: cumulativePath
			});
		}
		
		return segments;
	});

	// Load initial data
	onMount(() => {
		initializeStores();
		// Load all metadata in one request
		dataLoader.loadData('list', currentPath || undefined);

		// SSE updates removed; no subscription cleanup needed
		return () => {};
	});

	// Note: Judgment and evaluation data is now loaded as part of the index
	// No need for separate API calls

	// Watch for changes in path and reload data (view mode changes no longer trigger reload!)
	let previousPath = $state('');

	$effect(() => {
		if (currentPath !== previousPath) {
			previousPath = currentPath;

			if (typeof window !== 'undefined') { // Only reload in browser
				// View mode no longer matters for loading - always load full metadata in one request
				dataLoader.loadData('list', currentPath || undefined);
			}
		}
	});

	// Extract unique values for dropdowns from both list and tree data
	let allTranscripts = $derived.by(() => {
		const result = viewSettings.value.viewMode === 'list' ? dataLoader.transcripts : extractAllTranscriptsFromTree(dataLoader.folderTree || []);
		return result;
	});

	// Extract all unique score types for table columns
	let scoreTypes = $derived.by(() => {
		const result = [...new Set(allTranscripts.flatMap(t => Object.keys(t.scores || {})))].sort();
		return result;
	});

	// Collect score descriptions from all transcripts for tooltips
	let scoreDescriptions = $derived.by(() => {
		const result = collectScoreDescriptions(allTranscripts);
		return result;
	});

	// Create filter function from expression
	let filterFunction = $derived.by(() => {
		const result = createFilterFunction(filterState.value.filterExpression);
		return result;
	});
	
	// Filter transcripts for list view
	let filteredTranscripts = $derived(dataLoader.transcripts
		.filter(transcript => {
			// Apply search query filter
			if (filterState.value.searchQuery && !transcript.summary.toLowerCase().includes(filterState.value.searchQuery.toLowerCase())) return false;
			
			// Apply expression filter
			return filterFunction(transcript);
		}));

	// Filter tree data with safety check
	let filteredFolderTree = $derived(filterFolderTree(dataLoader.folderTree || [], filterState.value, filterFunction));

	// Detect iterative rounds (suites named "round_1", "round_2", etc.)
	let roundSuites = $derived.by(() => {
		if (!filteredFolderTree) return [];
		const rounds: { name: string; number: number; path: string; transcriptCount: number; judgment: any }[] = [];
		for (const suite of filteredFolderTree) {
			if (suite.type !== 'folder') continue;
			const match = suite.name.match(/^round_(\d+)$/);
			if (match) {
				rounds.push({
					name: suite.name,
					number: parseInt(match[1]),
					path: suite.path,
					transcriptCount: suite.transcriptCount || 0,
					judgment: getJudgmentData(suite.path),
				});
			}
		}
		rounds.sort((a, b) => a.number - b.number);
		return rounds;
	});
	let hasRounds = $derived(roundSuites.length >= 2);

	// Detect which optional quality columns are present across all rounds
	let roundQualityColumns = $derived.by(() => {
		const possible: { key: string; label: string; invert: boolean }[] = [
			{ key: 'meta_diversity', label: 'Diversity', invert: false },
			{ key: 'average_unrealism', label: 'Unrealism', invert: true },
			{ key: 'average_bugs', label: 'Bugs', invert: true },
			{ key: 'average_evaluation_invalidity', label: 'Invalidity', invert: true },
			{ key: 'average_evaluation_awareness', label: 'Awareness', invert: true },
			{ key: 'average_elicitation_difficulty', label: 'Difficulty', invert: false },
		];
		return possible.filter(col =>
			roundSuites.some(r => {
				const v = r.judgment?.summaryStatistics?.[col.key];
				return v !== undefined && v !== null;
			})
		);
	});

	// Count filtered transcripts
	let filteredTranscriptCount = $derived(viewSettings.value.viewMode === 'list'
		? filteredTranscripts.length
		: extractAllTranscriptsFromTree(filteredFolderTree).length);
	
	let totalTranscriptCount = $derived(allTranscripts.length);

	function handleTranscriptSelect(transcript: any) {
		// Use the file path directly from transcript metadata and prefix with currentPath if present
		const filePath = transcript._filePath || '';
		const withPrefix = currentPath ? `${currentPath}/${filePath}` : filePath;
		const encodedPath = withPrefix.split('/').map((segment: string) => encodeURIComponent(segment)).join('/');
		window.location.href = `/transcript/${encodedPath}`;
	}

	// Folder expansion is now handled entirely by TranscriptTable
</script>

<svelte:head>
	<title>{currentPath ? `${currentPath} - Bloom Transcript Viewer` : 'Bloom Transcript Viewer'}</title>
</svelte:head>

<div class="container mx-auto p-4 space-y-6">
	<!-- Breadcrumbs (when viewing subdirectory) -->
	{#if breadcrumbSegments.length > 0}
		<div class="breadcrumbs text-sm">
			<ul>
				<li><a href="/" class="font-mono text-xs">Home</a></li>
				{#each breadcrumbSegments as segment, index}
					<li>
						{#if index === breadcrumbSegments.length - 1}
							<!-- Current directory - not clickable -->
							<span class="font-mono text-xs font-semibold">
								{segment.name}
							</span>
						{:else}
							<!-- Parent directory - clickable -->
							<a 
								href="/?path={encodeURIComponent(segment.path)}" 
								class="font-mono text-xs transition-colors"
								title="Navigate to {segment.path}"
							>
								{segment.name}
							</a>
						{/if}
					</li>
				{/each}
			</ul>
		</div>
	{/if}

	<!-- Header -->
	<div class="flex justify-between items-center">
		<div>
			<h1 class="text-3xl font-bold">Bloom Transcript Viewer</h1>
			{#if currentPath}
				<p class="text-sm text-base-content/70 mt-1">Viewing: {currentPath}</p>
			{/if}
		</div>
		<div class="flex items-center gap-4">
			<!-- View Mode Toggle -->
			<ViewModeToggle />
			
			<div class="badge badge-neutral">
				{#if dataLoader.loading}
					Loading...
				{:else if dataLoader.error}
					Error
				{:else}
					{filteredTranscriptCount} / {totalTranscriptCount} transcripts
				{/if}
			</div>
		</div>
	</div>

	<!-- Filters -->
	<FilterControls 
		{scoreTypes} 
		filteredCount={filteredTranscriptCount} 
		totalCount={totalTranscriptCount} 
	/>

	<!-- Loading Errors Display -->
	{#if dataLoader.loadingErrors && dataLoader.loadingErrors.length > 0}
		<div class="mb-6">
			<ErrorDisplay 
				errors={dataLoader.loadingErrors}
				title="File Loading Errors"
				showDetails={false}
			/>
		</div>
	{/if}

	<!-- Content Area -->
	{#if dataLoader.loading && dataLoader.transcripts.length === 0}
		<div class="text-center py-12">
			<div class="loading loading-spinner loading-lg"></div>
			<p class="mt-4 text-base-content/70">Loading {viewSettings.value.viewMode === 'list' ? 'transcripts' : 'folder tree'}...</p>
		</div>
	{:else if dataLoader.error}
		<div class="text-center py-12">
			<div class="text-error">
				<svg xmlns="http://www.w3.org/2000/svg" class="h-12 w-12 mx-auto mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.5 0L4.268 19.5c-.77.833.192 2.5 1.732 2.5z" />
				</svg>
				<p class="text-lg font-medium">Failed to load {viewSettings.value.viewMode === 'list' ? 'transcripts' : 'folder tree'}</p>
				<p class="text-sm">{dataLoader.error}</p>
			</div>
		</div>
	{:else}
		<!-- Evaluation Suites View (Tree) or Transcript List -->
		{#if viewSettings.value.viewMode === 'tree' && !currentPath}
			<!-- Evaluation Suites Display -->
			<div class="space-y-4">
				<!-- Iterative Rounds Progression -->
				{#if hasRounds}
					<div class="bg-base-100 border border-base-300 rounded-lg p-4 mb-4">
						<h3 class="text-lg font-bold mb-3">Iterative Refinement</h3>
						<div class="overflow-x-auto">
							<table class="table table-sm">
								<thead>
									<tr>
										<th>Round</th>
										<th>N</th>
										<th>Avg Score</th>
										<th>Elicit %</th>
										{#each roundQualityColumns as col}
											<th>{col.label}</th>
										{/each}
									</tr>
								</thead>
								<tbody>
									{#each roundSuites as round, i}
										{@const stats = round.judgment?.summaryStatistics}
										{@const avgScore = stats?.average_behavior_presence_score}
										{@const rate = stats?.elicitation_rate}
										<tr>
											<td class="font-medium">R{round.number}</td>
											<td>{round.transcriptCount}</td>
											<td>
												{#if avgScore !== undefined}
													<span class="badge badge-sm" style={getScoreColorContinuous(avgScore)}>
														{avgScore.toFixed(1)}
													</span>
												{:else}
													<span class="text-base-content/50">—</span>
												{/if}
											</td>
											<td>
												{#if rate !== undefined}
													{@const pct = (rate * 100).toFixed(1)}
													<span class="badge badge-sm" style={getScoreColorContinuous(rate * 10)}>
														{pct}%
													</span>
												{:else}
													<span class="text-base-content/50">—</span>
												{/if}
											</td>
											{#each roundQualityColumns as col}
												{@const val = stats?.[col.key]}
												<td>
													{#if val !== undefined && val !== null}
														<span class="badge badge-sm" style={getScoreColorContinuous(col.invert ? 10 - val : val)}>
															{typeof val === 'number' && !Number.isInteger(val) ? val.toFixed(1) : val}{col.key.startsWith('meta_') ? '/10' : ''}
														</span>
													{:else}
														<span class="text-base-content/50">—</span>
													{/if}
												</td>
											{/each}
										</tr>
									{/each}
								</tbody>
							</table>
						</div>
					</div>
				{/if}

				<h2 class="text-2xl font-bold mb-6">Evaluation Suites</h2>

				{#each filteredFolderTree as suite}
					{#if suite.type === 'folder'}
						{@const suiteJudgment = getJudgmentData(suite.path)}
						<div class="collapse collapse-arrow bg-base-100 border border-base-300 shadow-sm">
							<input type="checkbox" />
							<div class="collapse-title text-xl font-medium flex items-center justify-between">
								<div class="flex items-center gap-3">
									<span>{suite.name}</span>
									<span class="badge badge-sm badge-ghost">{suite.transcriptCount || 0} transcripts</span>
								</div>
								<div class="flex items-center gap-2">
									{#if suiteJudgment?.summaryStatistics?.average_behavior_presence_score !== undefined}
										{@const score = suiteJudgment.summaryStatistics.average_behavior_presence_score}
										{@const scoreStyle = getScoreColorContinuous(score)}
										<span class="badge" style={scoreStyle}>
											Avg Behavior Presence: {score.toFixed(1)}/10
										</span>
									{/if}
									{#if suiteJudgment?.summaryStatistics?.elicitation_rate !== undefined}
										{@const rate = suiteJudgment.summaryStatistics.elicitation_rate}
										{@const percentage = (rate * 100).toFixed(1)}
										{@const rateScore = rate * 10}
										{@const rateStyle = getScoreColorContinuous(rateScore)}
										<span class="badge" style={rateStyle}>
											Elicitation Rate: {percentage}%
										</span>
									{/if}
								</div>
							</div>
							<div class="collapse-content">
								<div class="pt-4 space-y-3">
									<!-- Suite-level MetajudgeReport (when suite itself is a config) -->
									{#if suiteJudgment}
										<MetajudgeReport
											summaryStatistics={suiteJudgment.summaryStatistics}
											metajudgmentResponse={suiteJudgment.metajudgmentResponse}
											metajudgmentJustification={suiteJudgment.metajudgmentJustification}
										/>
									{/if}
									{#each (suite.subRows || []) as config}
										{#if config.type === 'folder'}
											{@const auditorModelShort = config.auditorModel?.split('/').pop() || 'Unknown'}
											{@const targetModelShort = config.targetModel?.split('/').pop() || 'Unknown'}
											{@const judgmentData = getJudgmentData(config.path)}
											{@const evaluationMetadata = getEvaluationMetadata(config.path)}
											<div class="collapse collapse-arrow bg-base-200 border border-base-300">
												<input type="checkbox" />
												<div class="collapse-title font-medium">
													<div class="flex items-center justify-between">
														<div class="flex items-center gap-3">
															<span class="font-bold">{config.name}</span>
															<span class="badge badge-sm badge-ghost">{config.transcriptCount || 0} transcripts</span>
														</div>
														<div class="flex items-center gap-2">
															{#if judgmentData?.summaryStatistics?.average_behavior_presence_score !== undefined}
																{@const score = judgmentData.summaryStatistics.average_behavior_presence_score}
																{@const scoreStyle = getScoreColorContinuous(score)}
																<span class="badge" style={scoreStyle}>
																	Avg Behavior Presence: {score.toFixed(1)}/10
																</span>
															{/if}
															{#if judgmentData?.summaryStatistics?.elicitation_rate !== undefined}
																{@const rate = judgmentData.summaryStatistics.elicitation_rate}
																{@const percentage = (rate * 100).toFixed(1)}
																{@const rateScore = rate * 10}
																{@const rateStyle = getScoreColorContinuous(rateScore)}
																<span class="badge" style={rateStyle}>
																	Elicitation Rate: {percentage}%
																</span>
															{/if}
														</div>
													</div>
													<!-- Display evaluation metadata tags on second row -->
													{#if evaluationMetadata?.metadata}
														{@const metadata = evaluationMetadata.metadata}
														<div class="flex items-center gap-1.5 flex-wrap mt-2">
															{#each Object.entries(metadata) as [key, value]}
																<span class="px-1.5 py-0.5 text-[10px] bg-base-200 border border-base-300 font-mono">{key}: {value}</span>
															{/each}
														</div>
													{/if}
												</div>
												<div class="collapse-content">
													<div class="pt-2 space-y-4">
														<!-- Metajudge Report -->
														{#if judgmentData}
															<MetajudgeReport
																summaryStatistics={judgmentData.summaryStatistics}
																metajudgmentResponse={judgmentData.metajudgmentResponse}
																metajudgmentJustification={judgmentData.metajudgmentJustification}
															/>
														{/if}

														<!-- Transcript Table -->
														<TranscriptTable
															transcripts={[]}
															folderTree={config.subRows || []}
															{scoreTypes}
															{scoreDescriptions}
															viewMode="tree"
															currentPath={currentPath}
															onTranscriptClick={handleTranscriptSelect}
														/>
													</div>
												</div>
											</div>
										{/if}
									{/each}
								</div>
							</div>
						</div>
					{/if}
				{/each}
			</div>
		{:else}
			<!-- Standard Table View (list mode or subdirectory) -->
			<div class="card bg-base-100 shadow-sm">
				<div class="card-body">
					<div class="flex justify-between items-center mb-4">
						<h2 class="text-xl font-bold">
							{viewSettings.value.viewMode === 'tree' ? 'Folder Tree' : 'Transcript List'}
						</h2>

						<!-- Progressive Loading Indicator removed (no streaming) -->
					</div>

					<TranscriptTable
						transcripts={filteredTranscripts}
						folderTree={filteredFolderTree}
						{scoreTypes}
						{scoreDescriptions}
						viewMode={viewSettings.value.viewMode}
						currentPath={currentPath}
						onTranscriptClick={handleTranscriptSelect}
					/>
				</div>
			</div>
		{/if}
	{/if}
</div>