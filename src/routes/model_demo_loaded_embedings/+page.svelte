<script lang="ts">
	import { onMount } from 'svelte';
	import * as tf from '@tensorflow/tfjs';
	import Npyjs from 'npyjs';

	let imageURL = '';
	let imgWidth = 1024; // Desired width for the canvas
	let imgHeight = 300; // Default height for the image
	let originalWidth: number = 0; // Original width of the image
	let originalHeight: number = 0; // Original height of the image
	let aspectRatio: number; // Aspect ratio of the image
	let clickedPositions: { x: number; y: number }[] = [];
	let embedding_tensor: tf.Tensor;
	interface modelInputInterface {
		image_embeddings: tf.Tensor;
		point_coords: tf.Tensor;
		point_labels: tf.Tensor;
		mask_input: tf.Tensor;
		has_mask_input: tf.Tensor;
		orig_im_size: tf.Tensor;
	}

	// let modelInput: any;
	let model: tf.GraphModel<string | tf.io.IOHandler>;
	let canvas: any;

	function handleCanvasClick(event: MouseEvent) {
		// const rect = canvas.getBoundingClientRect();
		// console.log("test")
		// console.log(event.offsetX, event.offsetY);

		
		//it logs -0 at 0,0 for some reason
		const x = Math.abs(event.offsetX);
		const y = Math.abs(event.offsetY);
		// const x = event.clientX - rect.left;
		// const y = event.clientY - rect.top;
		// console.log('Clicked at coordinates (x:', x, ', y:', y, ')');
		clickedPositions.push({ x, y });
		
		drawImageWithMarkers();

		// console.log(clickedPositions);
		// console.log(imgHeight)
		// console.log(imgWidth)
		// console.log(originalHeight)
		// console.log(originalWidth)
		console.log(x)
		console.log(y)
		console.log(x*(originalWidth/imgWidth))
		console.log(y*(originalHeight/imgHeight))
	}
	function drawImageWithMarkers() {
	
    // Clear canvas
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw image
    const img = new Image();
    img.src = imageURL;
    img.onload = () => {
      ctx.drawImage(img, 0, 0, imgWidth, imgHeight);
      
      // Draw markers
      ctx.fillStyle = 'green';
      for (const pos of clickedPositions) {
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 5, 0, Math.PI * 2);
        ctx.fill();
      }
    };
  }
	function undoLastClick() {
		// Remove last clicked position
		clickedPositions.pop();
		console.log(clickedPositions);

		// Redraw image with markers
		drawImageWithMarkers();
	}
	onMount(async () => {
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
		// Load the image from a static URL
		imageURL = './demobear.png';

		const img = new Image();
		img.src = imageURL;

		img.onload = () => {
			// Calculate aspect ratio
			aspectRatio = img.width / img.height;
			originalHeight = img.height;
			originalWidth = img.width;
			// Calculate height based on the aspect ratio and desired width
			imgHeight = imgWidth / aspectRatio;

			// Set canvas dimensions
			canvas.width = imgWidth;
			canvas.height = imgHeight;

			drawImageWithMarkers();

		};

		model = await tf.loadGraphModel('/tfjs_decoder_mobile/model.json');

		//load precomputed embeddings
		const npy = new Npyjs();
		const embeddings = await npy.load('image_embedding_tiny.npy');
		embedding_tensor = tf.tensor(embeddings.data as any, embeddings.shape, 'float32');

		// const response = await fetch(imageURL);
		// const blob = await response.blob();
		// const image = await createImageBitmap(blob);





		// // Get the original size of the image
		// const originalWidth = image.width;
		// const originalHeight = image.height;

		// Your input point and label data
		// const inputPoint = [
		// 	[600, 300],
		// 	[600, 400],
		// 	[500, 400],
		// 	[550, 450],
		// 	[0, 450],
		// 	[0, 0]
		// ];
		// const inputLabels = [1, 1, 1, 1, 1, -1];

		// // // Calculate scale factor
		// const longSideLength = 1024;
		// const scale = longSideLength / Math.max(originalWidth, originalHeight);

		// let newHeight = Math.round(originalHeight * scale);
		// let newWidth = Math.round(originalWidth * scale);

		// const realScaleWidth = newWidth / originalWidth;
		// const realScaleHeight = newHeight / originalHeight;
		// const onnxInputPoints = inputPoint.map(([x, y]) => [x * realScaleWidth, y * realScaleHeight]);

		// // // Create input tensors
		// const pointCoordsTensor = tf.tensor(
		// 	[onnxInputPoints],
		// 	[1, onnxInputPoints.length, 2],
		// 	'float32'
		// );
		// const pointLabelsTensor = tf.tensor([inputLabels], [1, inputLabels.length], 'float32');
		// const origImgSizeTensor = tf.tensor([originalHeight, originalWidth], [2], 'float32');
		// const maskInputTensor = tf.tensor(
		// 	new Float32Array(256 * 256).fill(0),
		// 	[1, 1, 256, 256],
		// 	'float32'
		// );
		// const hasMaskInputTensor = tf.tensor([0], [1], 'float32');

		// modelInput = {
		// 	image_embeddings: embeddings_tensor,
		// 	point_coords: pointCoordsTensor,
		// 	point_labels: pointLabelsTensor,
		// 	mask_input: maskInputTensor,
		// 	has_mask_input: hasMaskInputTensor,
		// 	orig_im_size: origImgSizeTensor
		// };
	});

	//TODO WIP
	async function createInputDict(){
		const inputPoint = clickedPositions;
		const inputLabels = clickedPositions.map(() => 1);

		// Calculate scale factor
		const longSideLength = 1024;
		const scale = longSideLength / Math.max(originalWidth, originalHeight);

		let newHeight = Math.round(originalHeight * scale);
		let newWidth = Math.round(originalWidth * scale);

		const realScaleWidth = newWidth / originalWidth;
		const realScaleHeight = newHeight / originalHeight;
		const onnxInputPoints = inputPoint.map(({x, y}) => [x * realScaleWidth, y * realScaleHeight]);

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


	async function runModel() {
		const modelInput = await createInputDict();
		// let data = modelInput.image_embeddings.arraySync();
		const predictions: any = await model.executeAsync(modelInput);
		const lastData = await predictions[predictions.length - 1].arraySync();

		let indices = [];
		const data = lastData[0][0];
		for (let i = 0; i < data.length; i++) {
			for (let j = 0; j < data[i].length; j++) {
				if (data[i][j] > 0.0) {
					indices.push([i, j]);
				}
			}
		}

		console.log(indices.length);
	}
</script>

<!-- Your image can be displayed here if needed -->
<div>
	<button on:click={runModel}>Run Model</button>
</div>
<canvas bind:this={canvas} on:click={handleCanvasClick}></canvas>
<button on:click={undoLastClick}>Undo</button>

<br />

<!-- <img src={imageURL} alt="Loaded" /> -->

<!-- {#if modelInput}
    <p>Model input loaded</p>
    <button on:click={runModel}>Run Model</button>
{/if} -->

<style>
	canvas {
		box-shadow: 0 0 10px rgba(0, 0, 0, 0.3);
		margin: 20px;
	}
</style>
