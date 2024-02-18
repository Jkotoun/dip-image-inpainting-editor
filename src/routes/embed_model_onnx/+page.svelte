<script lang="ts">
	import { onMount } from 'svelte';
	import * as tf from '@tensorflow/tfjs';
	import Npyjs from 'npyjs';
	import { Tensor } from 'onnxruntime-web';
	import { InferenceSession } from "onnxruntime-web";

	type ImageFile = {
		file: File;
		url: string;
	};


	let onnxSession: InferenceSession | null = null;
	let selectedImage: ImageFile | null = null;
	let resizedImageBlob: Blob | null = null;
	let imgTensor: tf.Tensor3D | null = null;

	async function resizeImage(file: File, longSideLength: number): Promise<Blob> {
    const image = await createImageBitmap(file);
    const canvas = document.createElement('canvas');

    let scale = longSideLength / Math.max(image.width, image.height);
    const newWidth = Math.round(image.width * scale);
    const newHeight = Math.round(image.height * scale);

    canvas.width = newWidth;
    canvas.height = newHeight;

    const ctx: any = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0, newWidth, newHeight);

    return new Promise((resolve, reject) => {
      canvas.toBlob((blob) => {
        if (!blob) {
          reject(new Error('Failed to resize image.'));
          return;
        }
        resolve(blob);
      }, file.type);
    });
  }

  async function loadImageTensor(imageBlob: Blob): Promise<tf.Tensor3D> {
    const imageElement = await createImageBitmap(imageBlob);
    let imageData = tf.browser.fromPixels(imageElement).toFloat();
    return imageData;
  }

  async function loadOnnxModel() {
    try {
        onnxSession = await InferenceSession.create("/mobile_sam.encoder.onnx", { 
          executionProviders: ['wasm'] ,
          graphOptimizationLevel: 'all'
        }
        );
        console.log('ONNX Model loaded successfully');
    } catch (error) {
        console.error('Error loading the ONNX model:', error);
    }
}

  async function handleFileInputChange(event: Event) {
    const files = (event.target as HTMLInputElement).files;

    if (!files || files.length === 0) return;

    const file = files[0];
    const url = URL.createObjectURL(file);
    selectedImage = { file, url };

    resizedImageBlob = await resizeImage(file, 1024);
    imgTensor = await loadImageTensor(resizedImageBlob);
	console.log(imgTensor)
  }

  async function runModel() {
    if (!imgTensor || !onnxSession) return;

    try {
        const input = new Float32Array(imgTensor.dataSync());
        const inputTensor = new Tensor('float32', input, imgTensor.shape);

		// const output = await onnxSession.run([[image_embeddings], {input_image: inputTensor}])
		onnxSession.run({input_image: inputTensor}).then((output) => {
      
			console.log(output);
		});
		

    } catch (error) {
        console.error('Error running the model:', error);
    }
}

  function downloadResizedImage() {
    if (!resizedImageBlob) return;

    const downloadLink = document.createElement('a');
    downloadLink.href = URL.createObjectURL(resizedImageBlob);
    downloadLink.download = 'resized_image.jpg';
    downloadLink.click();
  }

  

	// async function loadModels() {
	// 	try {
	// 		const imgEncoderPromise = tf.loadGraphModel('/enc_untransposed_tfjs/model.json');
			
	// 		const [imgEncoderValue] = await Promise.all([
	// 			imgEncoderPromise,
	// 		]);
	// 		SAMEncoder = imgEncoderValue;

	// 		isLoading = false; // Models loaded successfully
	// 	} catch (error) {
	// 		console.error('Error loading the model:', error);
	// 		isLoading = false; // Set loading state to false even on error
	// 	}
	// }

	// async function loadBackend() {
	// 	await tf.ready();
	// 	try {
	// 		await import('@tensorflow/tfjs-backend-webgpu');
	// 		await tf.setBackend('webgpu');
	// 		console.log(tf.env().getFlags());
	// 	} catch (e) {
	// 		try {
	// 			await tf.setBackend('webgl');
	// 		} catch (e) {
	// 			console.error('could not load backend:', e);
	// 			throw e;
	// 		}
	// 	}
	// }
	
	async function onLoad() {
    // Load backend and TFJS models
    // await loadBackend();

    // Load ONNX model
    await loadOnnxModel();
	isLoading = false;
}

	let SAMEncoder: tf.GraphModel<string | tf.io.IOHandler>;
	let embeddings: any;
	let isLoading = true;

	onMount(onLoad); // Load TFJS backend and models
</script>

{#if isLoading}
	<p>Loading model...</p>
{:else if onnxSession}
	<h1>AI Object remover</h1>

	<input type="file" accept="image/*" on:change={handleFileInputChange} />

  {#if selectedImage}
    <div>
      <h2>Original Image</h2>
      <img src={selectedImage.url} alt="placeholder" />
    </div>
  {/if}

  {#if resizedImageBlob}
    <div>
      <h2>Resized Image</h2>
      <img src={URL.createObjectURL(resizedImageBlob)} alt="placeholder" />
      <button on:click={downloadResizedImage}>Download Resized Image</button>
      <button on:click={runModel}>Run Model</button>
    </div>
  {/if}

	<h2>Image Encoder</h2>
<!-- 
	<ul>
		{#each SAMEncoder.inputs as input}
			<li>{input.name}, [{input.shape}], {input.dtype}</li>
		{/each}
	</ul> -->
{/if}

<style>
	/* Add your styles here */
</style>