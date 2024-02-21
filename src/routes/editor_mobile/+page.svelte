<script lang="ts">
	import { onMount } from 'svelte';
	import * as tf from '@tensorflow/tfjs';
	import Npyjs from 'npyjs';
	import { Tensor } from 'onnxruntime-web';
	import { InferenceSession } from 'onnxruntime-web';

	//models paths
	const mobileSAMEncoderPath = '/mobile_sam.encoder.onnx';
	const mobileSAMDecoderPath = '/tfjs_decoder_mobile/model.json';

	//types definition
	type pointType = 'positive' | 'negative';
	// interface modelInputInterface {
	// 	image_embeddings: tf.Tensor;
	// 	point_coords: tf.Tensor;
	// 	point_labels: tf.Tensor;
	// 	mask_input: tf.Tensor;
	// 	has_mask_input: tf.Tensor;
	// 	orig_im_size: tf.Tensor;
	// }

	//interface interaction globals
	let isLoading = true;
	let isEmbedderRunning = false;

	//segmentation model constants
	const longSideLength = 1024;
	let model: tf.GraphModel<string | tf.io.IOHandler>;
	let onnxSession: InferenceSession | null = null;
	let embedding_tensor: tf.Tensor;
	let resizedImageBlob: Blob | null = null;
	let resizedImgWidth: number;
	let resizedImgHeight: number;

	//editor globals - image and canvas interactions
	let imageURL: any;
	let uploadedImage: HTMLImageElement | null = null;
	let clickedPositions: { x: number; y: number; type: pointType }[] = [];
	let ImgResToCanvasSizeRatio: number = 1;
	let imageCanvas: any;
	let maskCanvas: any;
	let isPainting = false;
	let mask: boolean[][];
	//brush tool
	let brushSize = 10;
	let prevMouseX = 0;
	let prevMouseY = 0;
	let mouse: any;
	//EMBEDDING FUNCTIONS
	async function resizeImage(
		img: HTMLImageElement,
		imgFile: File,
		longSideLength: number = 1024
	): Promise<Blob> {
		//longerside
		console.log('current', img.width, img.height);
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
		const files = (event.target as HTMLInputElement).files;
		let uploadedImageFile: File = files?.[0] as File;
		if (!uploadedImageFile) return;

		//clean clicked positions
		clickedPositions = [];
		let reader = new FileReader();
		reader.onload = async (e) => {
			imageURL = e.target?.result as string;
			const img = new Image();
			img.src = imageURL as string;
			img.onload = async () => {
				// Calculate aspect ratio
				imageCanvas.width = img.width  
				// maskCanvas.width = img.width;
				imageCanvas.height = img.height;
				// maskCanvas.height = img.height;
				initializeMask(imageCanvas.width, imageCanvas.height);

				if (imageCanvas.width > imageCanvas.height) {
					imageCanvas.style.width = '100%';
					maskCanvas.style.width = '100%'

					imageCanvas.style.height = 'auto';
					maskCanvas.style.height = 'auto';
				} else {
					imageCanvas.style.width =  'auto';
					maskCanvas.style.width = 'auto';
					imageCanvas.style.height =  '70vh';
					maskCanvas.style.height =  '70vh';
				}
				const canvasElementSize = imageCanvas.getBoundingClientRect();
				ImgResToCanvasSizeRatio = img.width / canvasElementSize.width;

				console.log(imageCanvas.width, imageCanvas.height);
				drawImageWithMarkers(imageCanvas);
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
		isEmbedderRunning = true;
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
		let inputPoint: Array<{ x: number; y: number }> = clickedPositions.map(({ x, y }) =>
			coordsToResizedImgScale(x, y)
		);
		let inputLabels: Array<number> = clickedPositions.map(({ type }) =>
			type === 'positive' ? 1 : 0
		);

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
		//canvas is set to actual width and height on upload
		const origImgSizeTensor = tf.tensor([imageCanvas.height, imageCanvas.width], [2], 'float32');
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

		drawImageWithMask(data, imageCanvas);
	}

	//EDITOR HANDLING FUNCTIONS

	function coordsToResizedImgScale(x: number, y: number) {
		const imageX = (x / imageCanvas.width) * resizedImgWidth;
		const imageY = (y / imageCanvas.height) * resizedImgHeight;
		return { x: imageX, y: imageY };
	}

	function handleCanvasClick(event: MouseEvent) {
		
		//it logs -0 at 0,0 for some reason
		event.preventDefault();
		const xScaled = Math.abs(event.offsetX) * ImgResToCanvasSizeRatio;
		const yScaled = Math.abs(event.offsetY) * ImgResToCanvasSizeRatio;

		//left mouse button for positive, right for negative
		clickedPositions.push({
			x: xScaled,
			y: yScaled,
			type: event.button === 0 ? 'positive' : 'negative'
		});
		drawImageWithMarkers(imageCanvas);
	}
	function drawImageWithMarkers(canvas: HTMLCanvasElement) {
		// Clear canvas
		const ctx = canvas.getContext('2d');
		if (ctx) {
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
				for (const pos of clickedPositions) {
					ctx.fillStyle = pos.type === 'positive' ? 'green' : 'red';
					ctx.beginPath();
					ctx.arc(pos.x, pos.y, 5 * ImgResToCanvasSizeRatio, 0, Math.PI * 2);
					ctx.fill();
				}
			}
		}
	}

	function drawImageWithMask(mask: number[][], canvas: HTMLCanvasElement) {
		// Clear canvas
		// const ctx = canvas.getContext('2d');
		// ctx.clearRect(0, 0, canvas.width, canvas.height);
		drawImageWithMarkers(canvas);
		const ctx = canvas.getContext('2d');
		// Draw image
		if (uploadedImage && ctx) {
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
		drawImageWithMarkers(imageCanvas);
	}

	//brush tool
	// function startPainting(event: MouseEvent) {
	// 	isPainting = true;
	// 	mouse = getMousePos(canvas, event);

	// 	prevMouseX = mouse.x;
	// 	prevMouseY = mouse.y;
	// 	paint(event, canvas);
	// }

	// function stopPainting() {
	// 	isPainting = false;
	// }
	// function paint(event: MouseEvent, canvas: HTMLCanvasElement) {
	// 	if (isPainting) {
	// 		const x = event.offsetX;
	// 		const y = event.offsetY;
	// 		const xScaled = Math.abs(x) * ImgResToCanvasSizeRatio;
	// 		const yScaled = Math.abs(y) * ImgResToCanvasSizeRatio;
	// 		let ctx = canvas.getContext('2d');
	// 		// Draw on canvas
	// 		if (ctx) {
	// 			ctx.strokeStyle = 'rgba(89, 156, 255, 0.5)';
	// 			// ctx.fillRect((x - (brushSize / 2))*ImgResToCanvasSizeRatio, (y - (brushSize / 2))*ImgResToCanvasSizeRatio, brushSize, brushSize);
	// 			ctx.beginPath();
	// 			ctx.lineJoin = 'round';
	// 			ctx.lineWidth = brushSize;
	// 			ctx.moveTo(prevMouseX, prevMouseY);
	// 			ctx.lineTo(xScaled, yScaled);
	// 			//color
	// 			ctx.closePath();
	// 			ctx.stroke();
	// 			prevMouseX = xScaled;
	// 			prevMouseY = yScaled;
	// 		}
	// 	}
	// }

	function getMousePos(canvas: any, evt: any) {
		evt = evt.originalEvent || window.event || evt;
		var rect = canvas.getBoundingClientRect();

		if (evt.clientX !== undefined && evt.clientY !== undefined) {
			return {
				x: (evt.clientX - rect.left)*ImgResToCanvasSizeRatio,
				y: (evt.clientY - rect.top)*ImgResToCanvasSizeRatio
			};
		}
	}

	//Brush tool mask
	function startPainting(event: MouseEvent) {
		isPainting = true;
		mouse = getMousePos(maskCanvas, event);
		prevMouseX = mouse.x;
		prevMouseY = mouse.y;
		updateMask(prevMouseX, prevMouseY);
	}

	function paint(event: MouseEvent) {
		if (isPainting) {
			const x = event.offsetX*ImgResToCanvasSizeRatio;
			const y = event.offsetY*ImgResToCanvasSizeRatio;
			updateMaskWithBrush(x, y, brushSize);
			drawMask(maskCanvas.getContext('2d'));
			prevMouseX = x;
			prevMouseY = y;
		}
	}

	function stopPainting() {
		isPainting = false;
	}

	function drawMask(ctx: CanvasRenderingContext2D) {
		ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

		ctx.fillStyle = 'rgba(89, 156, 255, 0.5)';
		for (let y = 0; y < mask.length; y++) {
			for (let x = 0; x < mask[y].length; x++) {
				if (mask[y][x]) {
					ctx.fillRect(x, y, 1, 1);
				}
			}
		}
	}

	function initializeMask(canvasWidth: number, canvasHeight: number) {
		mask = new Array(canvasHeight);
		for (let i = 0; i < canvasHeight; i++) {
			mask[i] = new Array(canvasWidth).fill(false);
		}
	}

	// Function to update the mask array
	function updateMask(x: number, y: number) {
		if (x >= 0 && x < mask[0].length && y >= 0 && y < mask.length) {
			mask[Math.round(y)][Math.round(x)] = true;
		}
	}

	function updateMaskWithBrush(x: number, y: number, brushSize: number) {
    const radius = Math.floor((brushSize / 2)*ImgResToCanvasSizeRatio);
    for (let offsetY = -radius; offsetY <= radius; offsetY++) {
        for (let offsetX = -radius; offsetX <= radius; offsetX++) {
            const distanceSquared = offsetX * offsetX + offsetY * offsetY;
            if (distanceSquared <= radius * radius) {
                const pixelX = x + offsetX;
                const pixelY = y + offsetY;
                updateMask(pixelX, pixelY);
            }
        }
    }
}
	//ONMOUNT MODELS LOADINGS FUNCTIONS
	async function loadOnnxModel() {
		try {
			onnxSession = await InferenceSession.create(mobileSAMEncoderPath, {
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
		model = await tf.loadGraphModel(mobileSAMDecoderPath);
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

<div>
	<input type="file" accept="image/*" on:change={(e) => handleFileInputChange(e)} />
</div>
<div style="display: {uploadedImage ? 'block' : 'none'}">
	<button on:click={runModelDecoder} disabled={isEmbedderRunning}>Run Decoder</button>
	<button on:click={undoLastClick}>Undo</button>
	<!-- range slider to set brush size -->
	<label for="brushSize">Brush size: {brushSize}</label>
	<input type="range" min="1" max="100" bind:value={brushSize} />
	<div class="canvases">
		<canvas
		id="imageCanvas"
		bind:this={imageCanvas}
	
		/>
		<canvas
		id="maskCanvas"
		bind:this={maskCanvas}
		on:mousedown={startPainting}
		on:mouseup={stopPainting}
		on:mousemove={paint}
		/>
	</div>
</div>

<!-- TODO AI tool select -->
<!-- on:click={handleCanvasClick} -->
<!-- on:contextmenu={handleCanvasClick} -->

<style>
	.canvases {
		width: 60%;
		position: relative;
		
	}
	.canvases canvas{
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
		display:block;
		box-shadow: 0 0 10px 0 rgba(0, 0, 0, 0.2);
	}


</style>
