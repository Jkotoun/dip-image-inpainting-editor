<script lang="ts">
	import { onMount } from 'svelte';
	import * as tf from '@tensorflow/tfjs';
	import Npyjs from 'npyjs';
	import { Tensor } from 'onnxruntime-web';
	import { InferenceSession } from 'onnxruntime-web';

	let isLoading = true;
	let onnxSession: InferenceSession | null = null;
	let resizedImageBlob: Blob | null = null;
	let isEmbedderRunning = false;
	let uploadedImage: HTMLImageElement | null = null;
	let imageURL: any;
	let originalWidth: number = 0; // Original width of the image
	let originalHeight: number = 0; // Original height of the image
	let aspectRatio: number; // Aspect ratio of the image
	let resizedImgWidth: number;
	let resizedImgHeight: number;
	let clickedPositions: { x: number; y: number }[] = [];
	let embedding_tensor: tf.Tensor;
	// let embedding_tensor_loaded: tf.Tensor;

	interface modelInputInterface {
		image_embeddings: tf.Tensor;
		point_coords: tf.Tensor;
		point_labels: tf.Tensor;
		mask_input: tf.Tensor;
		has_mask_input: tf.Tensor;
		orig_im_size: tf.Tensor;
	}
	const longSideLength = 1024;
	// let modelInput: any;
	let model: tf.GraphModel<string | tf.io.IOHandler>;
	let canvas: any;

	//EMBEDDING FUNCTIONS

	async function resizeImage(
		img: HTMLImageElement,
		imgFile: File,
		longSideLength: number = 1024
	): Promise<Blob> {
		//longerside
		let newWidth, newHeight;
		if (img.width > img.height) {
			newWidth = longSideLength;
			newHeight = longSideLength * (img.height / img.width);
		} else {
			newHeight = longSideLength;
			newWidth = longSideLength * (img.width / img.height);
		}

		let tempCanvas = document.createElement('canvas');
		tempCanvas.width = newWidth;
		tempCanvas.height = newHeight;
		let tempContext = tempCanvas.getContext('2d');
		tempContext?.drawImage(img, 0, 0, newWidth, newHeight);
		resizedImgWidth = newWidth;
		resizedImgHeight = newHeight;

		console.log('new dims', newWidth, newHeight);
		return new Promise((resolve, reject) => {
			tempCanvas.toBlob((blob) => {
				if (!blob) {
					reject(new Error('Failed to resize image.'));
					return;
				}
				resolve(blob);
			}, imgFile.type);
		});
	}

	async function createImageTFTensor(
		uploadedImage: HTMLImageElement,
		uploadedImageFile: File,
		longSideLength: number = 1024
	): Promise<tf.Tensor3D> {
		resizedImageBlob = await resizeImage(uploadedImage, uploadedImageFile, longSideLength);
		const imageElement = await createImageBitmap(resizedImageBlob);
		let imageData = tf.browser.fromPixels(imageElement).toFloat();
		return imageData;
	}

	const handleFileInputChange = async (event: Event) => {
		isEmbedderRunning = true;
		const files = (event.target as HTMLInputElement).files;
		let uploadedImageFile: File = files?.[0] as File;
		if (!uploadedImageFile) return;

		let reader = new FileReader();
		reader.onload = async (e) => {
			imageURL = e.target?.result as string;
			const img = new Image();
			img.src = imageURL as string;
			img.onload = async () => {
				// Calculate aspect ratio
				aspectRatio = img.width / img.height;
				originalHeight = img.height;
				originalWidth = img.width;
				canvas.width = window.getComputedStyle(canvas).width.replace('px', '');
				canvas.height = canvas.width / aspectRatio;
				console.log(canvas.width, canvas.height);
				drawImageWithMarkers();
				const uploadedImgTFTensor = await createImageTFTensor(
					img,
					uploadedImageFile,
					longSideLength
				);
				if (uploadedImgTFTensor && onnxSession) {
					// isEmbedderRunning = true;
					const embedder_output = await runModelEncoder(onnxSession, uploadedImgTFTensor);
					if (embedder_output) {
						embedding_tensor = embedder_output;
						isEmbedderRunning = false;
					}
				}
			};
			uploadedImage = img;
		};
		reader.readAsDataURL(uploadedImageFile);
	};

	async function runModelEncoder(
		embedderOnnxSession: InferenceSession,
		imgTensor: tf.Tensor3D
	): Promise<tf.Tensor<tf.Rank> | undefined> {
		try {
			const input = new Float32Array(imgTensor.dataSync());
			const inputTensor = new Tensor('float32', input, imgTensor.shape);
			const output = await embedderOnnxSession.run({ input_image: inputTensor });
			return tf.tensor(
				output['image_embeddings'].data as any,
				output['image_embeddings'].dims as any,
				'float32'
			);
		} catch (error) {
			console.error('Error running the model:', error);
		}
		return;
	}

	//DECODER MODEL FUNCTIONS

	async function createInputDict() {
		const inputPoint = clickedPositions.map(({ x, y }) => convertCanvasToImageCoords(x, y));
		const inputLabels = clickedPositions.map(() => 1);

		const onnxInputPoints = inputPoint.map(({ x, y }) => [x, y]);

		//necessary when no box input is provided
		onnxInputPoints.push([0, 0]);
		inputLabels.push(-1);

		// Create input tensors
		const pointCoordsTensor = tf.tensor(
			[onnxInputPoints],
			[1, onnxInputPoints.length, 2],
			'float32'
		);
		const pointLabelsTensor = tf.tensor([inputLabels], [1, inputLabels.length], 'float32');
		const origImgSizeTensor = tf.tensor([originalHeight, originalWidth], [2], 'float32');
		const maskInputTensor = tf.tensor(
			new Float32Array(256 * 256).fill(0),
			[1, 1, 256, 256],
			'float32'
		);
		const hasMaskInputTensor = tf.tensor([0], [1], 'float32');

		let modelInput = {
			image_embeddings: embedding_tensor,
			point_coords: pointCoordsTensor,
			point_labels: pointLabelsTensor,
			mask_input: maskInputTensor,
			has_mask_input: hasMaskInputTensor,
			orig_im_size: origImgSizeTensor
		};
		return modelInput;
	}

	async function runModelDecoder() {
		const modelInput = await createInputDict();

		// let data = modelInput.image_embeddings.arraySync();
		const predictions: any = await model.executeAsync(modelInput);
		const lastData = await predictions[predictions.length - 1].arraySync();

		const data = lastData[0][0];

		drawImageWithMask(data);
	}

	//EDITOR HANDLING FUNCTIONS

	function convertCanvasToImageCoords(x: number, y: number) {
		const imageX = (x / canvas.width) * resizedImgWidth;
		const imageY = (y / canvas.height) * resizedImgHeight;
		return { x: imageX, y: imageY };
	}

	function handleCanvasClick(event: MouseEvent) {
		//it logs -0 at 0,0 for some reason
		const x = Math.abs(event.offsetX);
		const y = Math.abs(event.offsetY);
		clickedPositions.push({ x, y });
		console.log('original', x, y);
		console.log('converted', convertCanvasToImageCoords(x, y));
		drawImageWithMarkers();
	}
	function drawImageWithMarkers() {
		// Clear canvas
		const ctx = canvas.getContext('2d');
		ctx.clearRect(0, 0, canvas.width, canvas.height);
		if (uploadedImage) {
			ctx.drawImage(
				uploadedImage,
				0,
				0,
				uploadedImage.width,
				uploadedImage.height,
				0,
				0,
				canvas.width,
				canvas.height
			);
			ctx.fillStyle = 'green';
			for (const pos of clickedPositions) {
				ctx.beginPath();
				ctx.arc(pos.x, pos.y, 5, 0, Math.PI * 2);
				ctx.fill();
			}
		}
	}

	function drawImageWithMask(mask: number[][]) {
		// Clear canvas
		const ctx = canvas.getContext('2d');
		ctx.clearRect(0, 0, canvas.width, canvas.height);

		// Draw image
		if (uploadedImage) {
			ctx.drawImage(uploadedImage, 0, 0, canvas.width, canvas.height);

			ctx.fillStyle = 'green';
			for (const pos of clickedPositions) {
				ctx.beginPath();
				ctx.arc(pos.x, pos.y, 5, 0, Math.PI * 2);
				ctx.fill();
			}

			// Draw mask
			ctx.fillStyle = 'rgba(89, 156, 255, 0.5)';

			//iterate through canvas pixels and quant to mask values
			// let widthStep = canvas.width / resizedImgWidth;
			// let heightStep = canvas.height / resizedImgHeight;
			// for (let x = 0; x<canvas.width; x++) {
			// 	for (let y = 0; y<canvas.height; y++) {
			// 		if (mask[Math.round(y*heightStep)][Math.round(x*widthStep)] > 0) {
			// 			ctx.fillRect(x, y, 1, 1);
			// 		}
			// 	}
			// }



			const widthStep = canvas.width / mask[0].length;
			const heightStep = canvas.height / mask.length;
			for (let i = 0; i < mask.length; i++) {
				for (let j = 0; j < mask[i].length; j++) {
					if (mask[i][j] > 0) {
						ctx.fillRect(j * widthStep, i * heightStep, 1, 1);
					}
				}
			}
		}
	}

	function undoLastClick() {
		// Remove last clicked position
		clickedPositions.pop();

		// Redraw image with markers
		drawImageWithMarkers();
	}

	//ONMOUNT MODELS LOADINGS FUNCTIONS
	async function loadOnnxModel() {
		try {
			onnxSession = await InferenceSession.create('/mobile_sam.encoder.onnx', {
				executionProviders: ['wasm'],
				graphOptimizationLevel: 'all'
			});
			console.log('ONNX Model loaded successfully');
		} catch (error) {
			console.error('Error loading the ONNX model:', error);
		}
	}

	onMount(async () => {
		await tf.ready();
		try {
			await import('@tensorflow/tfjs-backend-webgpu');
			await tf.setBackend('webgpu');
			// console.log(tf.env().getFlags());
		} catch (e) {
			try {
				await tf.setBackend('webgl');
			} catch (e) {
				console.error('could not load backend:', e);
				throw e;
			}
		}

		await loadOnnxModel();
		model = await tf.loadGraphModel('/tfjs_decoder_mobile/model.json');
		isLoading = false;

		//LOADING EMBEDDINGS
		// //load precomputed embeddings
		// const npy = new Npyjs();
		// // const embeddings = await npy.load('image_embedding_tiny.npy');
		// const embeddings = await npy.load('large_res_img.npy');
		// // // console.log(embeddings.shape);
		// // // console.log(embeddings.data);

		// embedding_tensor_loaded = tf.tensor(embeddings.data as any, embeddings.shape, 'float32');
	});

	// Function to get the computed width of an element
	// function getComputedWidth(element: any) {
	// 	const style = ;
	// 	return style.width;
	// }

	// Wait for the window to load
</script>

{#if isLoading}
	<p>Loading model...</p>
{:else if isEmbedderRunning}
	<p>Running embedder...</p>
{:else}
	<p>All loaded</p>
{/if}

<h1>AI Object remover</h1>

<!-- <button on:click={() => isEmbedderRunning=true}></button> -->
<input type="file" accept="image/*" on:change={(e) => handleFileInputChange(e)} />

<!-- {#if uploadedImage} -->
<div>
	<!-- <button on:click={runModelEncoder}>Run Encoder</button> -->
	<button on:click={runModelDecoder} disabled={isEmbedderRunning}>Run Decoder</button>
</div>
<div id="canvasContainer">
	<canvas id="editorCanvas" bind:this={canvas} on:click={handleCanvasClick} />
</div>
<button on:click={undoLastClick}>Undo</button>

<button on:click={() => console.log()}>Log Canvas Width</button>

<!-- {/if} -->

<style>
	canvas {
		box-shadow: 0 0 10px rgba(0, 0, 0, 0.3);
		margin: 20px;
		width: 100%;
	}
	#canvasContainer {
		width: 60%;
	}
</style>
