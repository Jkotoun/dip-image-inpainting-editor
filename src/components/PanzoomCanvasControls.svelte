<script lang="ts">
	import { MinusIcon, MoveIcon, PlusIcon, ScanEyeIcon } from 'lucide-svelte';
	import { createEventDispatcher } from 'svelte';

	const dispatch = createEventDispatcher();
	export let disabled: boolean = false;
	export let currentZoomValue: number = 1;
	export let panEnabled: boolean = false;
</script>

<div class="btn-group variant-filled {disabled ? 'opacity-50 cursor-not-allowed' : ''}">
	<div class="flex items-center justify-center {disabled ? 'cursor-not-allowed' : ''}">
		<input
			type="checkbox"
			id="choose-me"
			class="peer hidden"
			{disabled}
			bind:checked={panEnabled}
		/>

		<label
			for="choose-me"
			class="select-none {disabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'} rounded-lg
      peer-checked:bg-surface-700 dark:peer-checked:bg-surface-200 peer-checked:border-gray-200 !px-1 !mx-1 !ml-2 lg:!p-2 lg:!mr-2 lg:!ml-3"
		>
			<MoveIcon />
		</label>
	</div>
	<button on:click={(e) => e.preventDefault()} {disabled} class="cursor-default !px-1 lg:!px-3">
		<div class="flex items-center justify-center gap-x-2">
			<button
				on:click={(e) => dispatch('zoomOut')}
				class="{disabled
					? 'cursor-not-allowed'
					: 'cursor-pointer'} btn btn-sm !p-0 lg:!px-4 lg:!py-2"
			>
				<MinusIcon />
			</button>

			<span
				class="border-2 lg:text-md text-sm font-medium rounded-md p-2 border-surface-600 dark:border-surface-700"
				>{Math.round(currentZoomValue * 100)} %</span
			>
			<button
				on:click={(e) => dispatch('zoomIn')}
				class="{disabled
					? 'cursor-not-allowed'
					: 'cursor-pointer'} btn btn-sm variant-filled !p-0 lg:!px-4 lg:!py-2"
			>
				<PlusIcon />
			</button>
		</div>
	</button>
	<button {disabled} on:click={() => dispatch('zoomReset')} class="!px-2 !pr-3 lg:!px-4 lg:!pr-5">
		<ScanEyeIcon />
	</button>
</div>

<style>
</style>
