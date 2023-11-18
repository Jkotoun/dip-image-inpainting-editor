<script lang="ts">
	import { onMount } from 'svelte';
	import * as tf from '@tensorflow/tfjs';
  //TODO move loadBackend and loadModels to some lib file
	async function loadBackend() {
		await tf.ready();
		try {
			await import('@tensorflow/tfjs-backend-webgpu');
			await tf.setBackend('webgpu');
			console.log(tf.env().getFlags());
		} catch (e) {
			try {
				await tf.setBackend('webgl');
			} catch (e) {
				console.error('could not load backend:', e);
				throw e;
			}
		}
		await loadModels();
	}

  async function loadModels() {
		try {
			const imgEncoderPromise = tf.loadGraphModel('/mobile_enc_tfjs/model.json');
			const decoderPromise = tf.loadGraphModel('/base_dec_tfjs/model.json');
			const wavepaintPromise = tf.loadGraphModel('/wavepaint_tfjs_256/model.json');
			const [imgEncoderValue, decoderValue, wavepaintValue] = await Promise.all([
				imgEncoderPromise,
				decoderPromise,
				wavepaintPromise
			]);
			wavepaint = wavepaintValue;
			SAMDecoder = decoderValue;
			SAMEncoder = imgEncoderValue;
			isLoading = false; // Models loaded successfully
		} catch (error) {
			console.error('Error loading the model:', error);
			isLoading = false; // Set loading state to false even on error
		}
	}

	async function onLoad() {
		try {
			await loadBackend();
			console.log(tf.getBackend(), 'backend loaded'); //should be webgpu
		} catch (e) {
      isLoading = false;
			console.error('could not load backend:', e);
		}
		try {
			await loadModels();
			console.log('models loaded');
			console.log('enc', SAMEncoder);
			console.log('dec', SAMDecoder);
			console.log('wave', wavepaint);
		} catch (e) {
      isLoading = false;
			console.error('could not load models:', e);
		}
	}

	let SAMEncoder: tf.GraphModel<string | tf.io.IOHandler>;
	let SAMDecoder: tf.GraphModel<string | tf.io.IOHandler>;
	let wavepaint: tf.GraphModel<string | tf.io.IOHandler>;
	let isLoading = true;
	
	onMount(onLoad); // Load TFJS backend and models
</script>

{#if isLoading}
	<p>Loading model...</p>
{:else if SAMEncoder && SAMDecoder && wavepaint}
	<h1>AI Object remover</h1>
	<h2>Image Encoder</h2>
	<ul>
		{#each SAMEncoder.inputs as input}
			<li>{input.name}, [{input.shape}], {input.dtype}</li>
		{/each}
	</ul>
	<h2>Promt Decoder</h2>
	<ul>
		{#each SAMDecoder.inputs as input}
			<li>{input.name}, [{input.shape}], {input.dtype}</li>
		{/each}
	</ul>
	<h2>Wavepaint</h2>
	<ul>
		{#each wavepaint.inputs as input}
			<li>{input.name}, [{input.shape}], {input.dtype}</li>
		{/each}
	</ul>
{/if}

<style>
	/* Add your styles here */
</style>
