<!-- <script lang="ts">
	import { onMount } from 'svelte';
	import { InferenceSession } from 'onnxruntime-web';
	import * as tf from '@tensorflow/tfjs';

	type ImageFile = {
		file: File;
		url: string;
	};

	let selectedImage: ImageFile | null = null;
	let resizedImageBlob: Blob | null = null;
	let imgTensor: tf.Tensor3D | null = null;
	let session: InferenceSession | null = null;
	let isLoading = true;

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
		const imageData = tf.browser.fromPixels(imageElement).toFloat();
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
		console.log(imgTensor);
	}

	function downloadResizedImage() {
		if (!resizedImageBlob) return;

		const downloadLink = document.createElement('a');
		downloadLink.href = URL.createObjectURL(resizedImageBlob);
		downloadLink.download = 'resized_image.jpg';
		downloadLink.click();
	}

	async function runModel() {
		if (!imgTensor || !session) return;

		const inputTensor = tf.cast(imgTensor, 'float32'); // Convert to float32
		const inputArrayBuffer = inputTensor.dataSync().buffer; // Convert to ArrayBuffer
		const inputTensorSize = inputTensor.size;

		const outputTensor = await onnxSession.run([new Float32Array(inputArrayBuffer)], {
			input: [1, 3, 1024, 1024]
		}); // Assuming input shape is [1, 3, 1024, 1024]
		console.log(outputTensor);

		// // Assuming you have loaded your model as SAMEncoder
		// const output = SAMEncoder.predict(imgTensor) as tf.Tensor;
		// console.log(output);
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
			// const imgEncoderPromise = tf.loadGraphModel('/enc_untransposed_tfjs/model.json');

			// const decoderPromise = tf.loadGraphModel('/base_dec_tfjs/model.json');
			// const decoder2Promise = tf.loadGraphModel('/mobilesam_base_decoder_tfjs/model.json');

			// const wavepaintPromise = tf.loadGraphModel('/wavepaint_tfjs_256/model.json');
			// const [imgEncoderValue, decoderValue, wavepaintValue, decoder2Value] = await Promise.all([
			// 	imgEncoderPromise,
			// 	decoderPromise,
			// 	wavepaintPromise,
			// 	decoder2Promise
			// ]);
			// // wavepaint = wavepaintValue;
			// // SAMDecoder = decoderValue;
			// // SAMDecoder2 = decoder2Value;
			// SAMEncoder = imgEncoderValue;
			session = await InferenceSession.create('/mobile_sam.encoder.quant.onnx'); // Change path to your ONNX model

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
			// console.log('enc', SAMEncoder);
			// console.log('dec', SAMDecoder);
			// console.log('wave', wavepaint);
		} catch (e) {
			isLoading = false;
			console.error('could not load models:', e);
		}
	}

	// let SAMEncoder: tf.GraphModel<string | tf.io.IOHandler>;
	// let SAMDecoder: tf.GraphModel<string | tf.io.IOHandler>;
	// let SAMDecoder2: tf.GraphModel<string | tf.io.IOHandler>;

	// let wavepaint: tf.GraphModel<string | tf.io.IOHandler>;

	onMount(onLoad); // Load TFJS backend and models
</script>

{#if isLoading}
	<p>Loading model...</p>
{:else if session}
	<!-- // && SAMDecoder && wavepaint && SAMDecoder2 -->

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
	<!-- <h2>Promt Encoder / Mask Decoder</h2>
	<ul>
		{#each SAMDecoder.inputs as input}
			<li>{input.name}, [{input.shape}], {input.dtype}</li>
		{/each}
	</ul>
	<h2>Promt Encoder / Mask Decoder 2</h2>
	<ul>
		{#each SAMDecoder2.inputs as input}
			<li>{input.name}, [{input.shape}], {input.dtype}</li>
		{/each}
	</ul>
	<h2>Wavepaint</h2>
	<ul>
		{#each wavepaint.inputs as input}
			<li>{input.name}, [{input.shape}], {input.dtype}</li>
		{/each}
	</ul> -->
{/if}

<style>
	/* Add your styles here */
</style> -->
