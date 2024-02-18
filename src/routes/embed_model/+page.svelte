<script lang="ts">
	import { onMount } from 'svelte';
	import * as tf from '@tensorflow/tfjs';
	import Npyjs from 'npyjs';

	type ImageFile = {
		file: File;
		url: string;
	};



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

  function downloadResizedImage() {
    if (!resizedImageBlob) return;

    const downloadLink = document.createElement('a');
    downloadLink.href = URL.createObjectURL(resizedImageBlob);
    downloadLink.download = 'resized_image.jpg';
    downloadLink.click();
  }

  async function runModel() {
	  if (!imgTensor) return;
    console.log(imgTensor)
	console.log(imgTensor.shape)


	console.log(imgTensor.arraySync().at(0)?.at(0)?.at(0))
	console.log(typeof(imgTensor.arraySync().at(0)?.at(0)?.at(0)))
    // Assuming you have loaded your model as SAMEncoder
    const output = SAMEncoder.predict(imgTensor) as tf.Tensor;
    console.log(output);
  }
	//TODO move loadBackend and loadModels to some lib
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
			const imgEncoderPromise = tf.loadGraphModel('/enc_untransposed_tfjs/model.json');
			
			const [imgEncoderValue] = await Promise.all([
				imgEncoderPromise,
			]);
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
			// const npy = new Npyjs();
			// embeddings =  await npy.load('embedding_onnx.npy');
			
			console.log(embeddings)
			console.log(tf.getBackend(), 'backend loaded'); //should be webgpu

		} catch (e) {
			isLoading = false;
			console.error('could not load backend:', e);
		}
		try {
			await loadModels();
			console.log('models loaded');
			console.log('enc', SAMEncoder);
		} catch (e) {
			isLoading = false;
			console.error('could not load models:', e);
		}
	}

	let SAMEncoder: tf.GraphModel<string | tf.io.IOHandler>;
	let embeddings: any;
	let isLoading = true;

	onMount(onLoad); // Load TFJS backend and models
</script>

{#if isLoading}
	<p>Loading model...</p>
{:else if SAMEncoder}
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

	<ul>
		{#each SAMEncoder.inputs as input}
			<li>{input.name}, [{input.shape}], {input.dtype}</li>
		{/each}
	</ul>
{/if}

<style>
	/* Add your styles here */
</style>