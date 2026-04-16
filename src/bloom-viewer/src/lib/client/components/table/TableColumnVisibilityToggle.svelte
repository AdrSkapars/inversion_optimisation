<script lang="ts">
  import type { VisibilityState } from '@tanstack/svelte-table';

  interface Props {
    columnVisibility: VisibilityState;
    allColumns: { id: string; header: string }[];
    onToggle: (columnId: string, visible: boolean) => void;
  }

  let { columnVisibility, allColumns, onToggle }: Props = $props();

  function handleToggle(columnId: string) {
    const currentVisibility = columnVisibility[columnId] ?? true;
    onToggle(columnId, !currentVisibility);
  }
</script>

<div class="dropdown dropdown-end">
  <div tabindex="0" role="button" class="btn btn-sm btn-outline">
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-4 h-4">
      <path stroke-linecap="round" stroke-linejoin="round" d="M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178z" />
      <path stroke-linecap="round" stroke-linejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
    </svg>
    Columns
  </div>
  <div class="dropdown-content z-[1] menu p-2 shadow bg-base-100 rounded-box w-52">
    <div class="form-control">
      <label class="label cursor-pointer">
        <span class="label-text font-medium">Toggle All</span>
        <input 
          type="checkbox" 
          class="toggle toggle-sm" 
          checked={allColumns.every(col => columnVisibility[col.id] !== false)}
          onchange={(e) => {
            const checked = e.currentTarget.checked;
            allColumns.forEach(col => onToggle(col.id, checked));
          }}
        />
      </label>
    </div>
    <div class="divider my-1"></div>
    {#each allColumns as column}
      <div class="form-control">
        <label class="label cursor-pointer">
          <span class="label-text text-sm">{column.header}</span>
          <input 
            type="checkbox" 
            class="checkbox checkbox-sm" 
            checked={columnVisibility[column.id] !== false}
            onchange={() => handleToggle(column.id)}
          />
        </label>
      </div>
    {/each}
  </div>
</div>