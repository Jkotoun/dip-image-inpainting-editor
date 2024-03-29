<script lang="ts">
	type tool = 'brush' | 'segment_anything';
    type brushMode = 'brush' | 'eraser';
    type SAMMode = 'positive' | 'negative';
    export let selectedTool: tool = 'brush';
    export let selectedBrushMode: brushMode = 'brush';
    export let selectedSAMMode: SAMMode = 'positive';
    export let SAMMaskDilatation : number = 10;
    export let brushToolSize : number = 10;
</script>


<style>

</style>