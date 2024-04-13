<script lang="ts">
	import { RadioGroup, RadioItem, Tab, TabGroup } from '@skeletonlabs/skeleton';
	import type { tool, brushMode, SAMMode } from '../types/editorTypes';
	import { Brush, Eraser, MinusCircle, PlusCircleIcon, WandSparkles } from 'lucide-svelte';
	export let selectedTool: tool = 'segment_anything';
	export let selectedBrushMode: brushMode = 'brush';
	export let selectedSAMMode: SAMMode = 'positive';
	export let SAMMaskDilatation: number = 10;
	export let brushToolSize: number = 10;
</script>

<TabGroup>
	<Tab class="px-8 py-4" bind:group={selectedTool} name="segment_anything" value="segment_anything">
		<span class="flex gap-x-2 items-center"> <WandSparkles size={18} /> Smart selector</span>
	</Tab>
	<Tab class="px-8 py-4" bind:group={selectedTool} name="brush" value="brush">
		<span class="flex gap-x-2 items-center"> <Brush size={18} /> Brush</span>
	</Tab>
	
	<div slot="panel" class="px-4 py-2">

		{#if selectedTool === 'brush'}
			<div>
				<label for="brushtoolselect" class="font-semibold">Select tool:</label>
				<div class="font-thin">Select brush or eraser tool to mark the area you want to remove</div>
				<RadioGroup class="text-token mb-4" id="brushtoolselect">
					<RadioItem bind:group={selectedBrushMode} name="brushtool" value="brush">
						<Brush size={18} />
					</RadioItem>
					<RadioItem bind:group={selectedBrushMode} name="brushtool" value="eraser">
						<Eraser size={18} />
					</RadioItem>
				</RadioGroup>

				<label for="brushSize">Brush size: {brushToolSize}</label>
				<input type="range" min="1" max="500" bind:value={brushToolSize} />
			</div>
		{:else if selectedTool === 'segment_anything'}

			<label for="sammodeselect" class="font-semibold">Select smart selector mode:</label>
			<div class="font-thin">
				Add positive (adds area to selection) or negative (removes area from selection) points for
				selection
			</div>
			<RadioGroup id="sammodeselect" class="text-token mb-4">
				<RadioItem bind:group={selectedSAMMode} name="sammode" value="positive">
					<PlusCircleIcon size={18} />
				</RadioItem>
				<RadioItem bind:group={selectedSAMMode} name="sammode" value="negative">
					<MinusCircle size={18} />
				</RadioItem>
			</RadioGroup>
			<label for="pixelsDilatation">Mask dilatation: {SAMMaskDilatation}</label>
			<input type="range" min="0" max="25" bind:value={SAMMaskDilatation} />
		{/if}
		<div class="font-thin text-sm py-3">Tip: For best results, selected area should contain all edges and shadows of the object.</div>

	</div>
</TabGroup>

<style>
</style>
