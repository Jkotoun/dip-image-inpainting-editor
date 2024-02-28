<script lang="ts">
	import { onMount } from 'svelte';
	import * as tf from '@tensorflow/tfjs';
	import { Tensor } from 'onnxruntime-web';
	import { InferenceSession } from 'onnxruntime-web';

	//models paths
	const mobileSAMEncoderPath = '/mobile_sam.encoder.onnx';
	const mobileSAMDecoderPath = '/tfjs_decoder_mobile/model.json';
	const mobile_inpainting_GAN = '/migan_pipeline_v2.onnx';

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
	interface loadedImgRGBData {
		rgbArray: Uint8Array;
		width: number;
		height: number;
	}

	let isLoading = true;
	let isEmbedderRunning = false;

	//segmentation model constants
	const longSideLength = 1024;
	let model: tf.GraphModel<string | tf.io.IOHandler>;
	let onnxSession: InferenceSession | null = null;
	let onnxSessionMIGAN: InferenceSession | null = null;
	let embedding_tensor: tf.Tensor;
	let resizedImgWidth: number;
	let resizedImgHeight: number;
	// let loadedImgRGB: loadedImgRGBData;

	//inpaint model constants
	let inpaintedImgCanvas: any;

	//editor globals - image and canvas interactions
	let imagebase64URL: any;
	let uploadedImage: HTMLImageElement | null = null;
	interface SAMmarker {
		x: number;
		y: number;
		type: pointType;
	}
	let clickedPositions: SAMmarker[] = [];
	let ImgResToCanvasSizeRatio: number = 1;
	let imageCanvas: any;
	let maskCanvas: any;
	let isPainting = false;
	// let mask: boolean[][];
	let masksStatesHistoryStack: any[] = [];
	let masksStatesUndoedStack: any[] = [];

	//editor state
	//refactored undo prep
	interface editorState {
		maskBrush: boolean[][];
		maskSAM: boolean[][];
		clickedPositions: { x: number; y: number; type: pointType }[];
		imgData: ImageData;
	}
	let imgDataOriginal: ImageData;

	let currentCursor: 'default' | 'brush' | 'eraser' = 'default';

	//brush tool
	let brushSize = 10;
	let prevMouseX = 0;
	let prevMouseY = 0;
	let currentCanvasRelativeX = 0;
	let currentCanvasRelativeY = 0;
	// let mouse: any;
	let selectedBrushMode: 'brush' | 'eraser' = 'brush'; // Initial selected option
	let selectedTool: 'brush' | 'segment_anything' = 'segment_anything';
	//EMBEDDING FUNCTIONS
	function getResizedImgRGBArray(
		img: HTMLImageElement,
		longSideLength: number = 1024
	): loadedImgRGBData {
		//longerside
		let newWidth, newHeight;
		if (img.width > img.height) {
			newWidth = longSideLength;
			newHeight = Math.round(longSideLength * (img.height / img.width));
		} else {
			newHeight = longSideLength;
			newWidth = Math.round(longSideLength * (img.width / img.height));
		}

		let tempCanvas = document.createElement('canvas');
		tempCanvas.width = newWidth;
		tempCanvas.height = newHeight;
		let tempContext = tempCanvas.getContext('2d');
		tempContext?.drawImage(img, 0, 0, newWidth, newHeight);
		resizedImgWidth = newWidth;
		resizedImgHeight = newHeight;
		let tmpCanvasData = getImageData(tempCanvas);
		return imgDataToRGBArray(tmpCanvasData);
	}

	function imgDataToRGBArray(imgData: ImageData): loadedImgRGBData {
		// let canvasContext = canvas.getContext('2d');
		// let imgData = canvasContext?.getImageData(0, 0, canvas.width, canvas.height);
		let pixels = imgData?.data;
		//create rgb array
		let rgbArray = new Uint8Array(imgData.width * imgData.height * 3);
		for (let i = 0; i < imgData.width * imgData.height; i++) {
			rgbArray[i * 3] = pixels![i * 4];
			rgbArray[i * 3 + 1] = pixels![i * 4 + 1];
			rgbArray[i * 3 + 2] = pixels![i * 4 + 2];
		}
		return { rgbArray, width: imgData.width, height: imgData.height } as loadedImgRGBData;
	}

	const handleFileInputChange = async (event: Event) => {
		const files = (event.target as HTMLInputElement).files;
		let uploadedImageFile: File = files?.[0] as File;
		if (!uploadedImageFile) return;

		//clean clicked positions
		clickedPositions = [];
		let reader = new FileReader();
		reader.onload = async (e) => {
			imagebase64URL = e.target?.result as string;
			const img = new Image();
			img.src = imagebase64URL as string;
			img.onload = async () => {
				// Calculate aspect ratio
				imageCanvas.width = img.width;
				maskCanvas.width = img.width;
				imageCanvas.height = img.height;
				maskCanvas.height = img.height;
				// initializeMask(imageCanvas.width, imageCanvas.height);

				if (imageCanvas.width > imageCanvas.height) {
					imageCanvas.style.width = maskCanvas.style.width = '100%';
					imageCanvas.style.height = maskCanvas.style.height = 'auto';
				} else {
					imageCanvas.style.width = maskCanvas.style.width = 'auto';
					imageCanvas.style.height = maskCanvas.style.height = '70vh';
				}
				const canvasElementSize = imageCanvas.getBoundingClientRect();
				ImgResToCanvasSizeRatio = img.width / canvasElementSize.width;

				const ctx = imageCanvas.getContext('2d');
				if (uploadedImage) {
					ctx.drawImage(
						uploadedImage,
						0,
						0,
						uploadedImage.width,
						uploadedImage.height,
						0,
						0,
						imageCanvas.width,
						imageCanvas.height
					);
				imgDataOriginal = getImageData(imageCanvas);
				}

				// drawImage(imageCanvas);
				drawMarkers(imageCanvas, clickedPositions);

				//input for SAM encoder
				let uploadedResizedImgRGBData = await getResizedImgRGBArray(img, longSideLength);
				if (uploadedResizedImgRGBData && onnxSession) {
					// isEmbedderRunning = true;
					const embedder_output = await runModelEncoder(onnxSession, uploadedResizedImgRGBData);

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
		imgRGBData: loadedImgRGBData
	): Promise<tf.Tensor<tf.Rank> | undefined> {
		try {
			// const input = new Float32Array(imgTensor.dataSync());
			let floatArray = Float32Array.from(imgRGBData.rgbArray);
			const inputTensor = new Tensor('float32', floatArray, [
				imgRGBData.height,
				imgRGBData.width,
				3
			]);
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

		//convert to bool array
		const maskArray = data.map((val: number[]) => val.map((v) => v > -10.0));

		//combine maskArray with current state mask with or
		let maskArrayCombined;
		if(masksStatesHistoryStack.length > 0){
			maskArrayCombined = maskArray.map((row: boolean[], y: number) =>
				row.map((val, x) => val || masksStatesHistoryStack[masksStatesHistoryStack.length - 1][y][x])
			);
		} else {
			maskArrayCombined = maskArray;
		}

		drawMask(maskCanvas, maskArrayCombined);
		masksStatesHistoryStack = [...masksStatesHistoryStack, maskArrayCombined];
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
		//clear canvas
		clearCanvas(imageCanvas);
		drawImage(imageCanvas, imgDataOriginal);
		drawMarkers(imageCanvas, clickedPositions);
	}
	// function drawImage(canvas: HTMLCanvasElement, imageData: ImageData) {
		// Clear canvas
		// const ctx = canvas.getContext('2d');
		// if (ctx) {
		// 	ctx.clearRect(0, 0, canvas.width, canvas.height);
		// 	console.log(uploadedImage)
		// 	if (uploadedImage) {
		// 		ctx.drawImage(
		// 			uploadedImage,
		// 			0,
		// 			0,
		// 			uploadedImage.width,
		// 			uploadedImage.height,
		// 			0,
		// 			0,
		// 			canvas.width,
		// 			canvas.height
		// 		);
		// 	}
		// }
	// }

		function drawImage(canvas: HTMLCanvasElement, imageData: ImageData) {
			console.log("drawing")
			console.log(imageData)
			canvas.getContext('2d')!.putImageData(imageData, 0, 0);
		}	


	function drawMarkers(canvas: HTMLCanvasElement, clickedPositions: SAMmarker[]) {
		const canvasContext = canvas.getContext('2d');
		if (!canvasContext) return;
		for (const pos of clickedPositions) {
			canvasContext.fillStyle = pos.type === 'positive' ? 'green' : 'red';
			canvasContext.beginPath();
			canvasContext.arc(pos.x, pos.y, 5 * ImgResToCanvasSizeRatio, 0, Math.PI * 2);
			canvasContext.fill();
		}
	}

	function undoLastAction() {
		//check before removing, if there is any
		if (masksStatesHistoryStack.length > 0) {
			masksStatesUndoedStack = [
				...masksStatesUndoedStack,
				masksStatesHistoryStack[masksStatesHistoryStack.length - 1]
			];
			masksStatesHistoryStack = masksStatesHistoryStack.slice(0, -1);
		}
		// after remove, check if there is state to draw to canvas
		if (masksStatesHistoryStack.length > 0) {
			drawMask(maskCanvas, masksStatesHistoryStack[masksStatesHistoryStack.length - 1]);
		} else {
			clearCanvas(maskCanvas);
		}
		// Redraw image with markers
		//TODO refactor
		drawImage(imageCanvas, imgDataOriginal);
		drawMarkers(imageCanvas, clickedPositions);
	}

	function redoLastAction() {
		if (masksStatesUndoedStack.length > 0) {
			masksStatesHistoryStack = [
				...masksStatesHistoryStack,
				masksStatesUndoedStack[masksStatesUndoedStack.length - 1]
			];
			masksStatesUndoedStack = masksStatesUndoedStack.slice(0, -1);
			// drawMask(maskCanvas, masksStatesHistoryStack[masksStatesHistoryStack.length - 1]);
			// drawImageWithMarkers(imageCanvas);
		}
	}

	// brush tool
	function startPainting(event: MouseEvent) {
		isPainting = true;
		prevMouseX = event.offsetX * ImgResToCanvasSizeRatio;
		prevMouseY = event.offsetY * ImgResToCanvasSizeRatio;
		handleEditorMouseMove(event, maskCanvas);
	}

	function stopPainting() {
		isPainting = false;
		let maskArray = createMaskArray(maskCanvas);
		masksStatesHistoryStack = [...masksStatesHistoryStack, maskArray];
		masksStatesUndoedStack = [];
		// drawImageWithMarkers(imageCanvas);
		// drawMask(maskCanvas, masksStatesStack[masksStatesStack.length - 1]);
	}
	function handleEditorMouseMove(event: MouseEvent, canvas: HTMLCanvasElement) {
		const x = event.offsetX * ImgResToCanvasSizeRatio;
		const y = event.offsetY * ImgResToCanvasSizeRatio;

		currentCanvasRelativeX = event.offsetX;
		currentCanvasRelativeY = event.offsetY;
		if (isPainting) {
			paintOnCanvas(x, y, brushSize, canvas);
		}
	}

	function showBrushCursor(event: MouseEvent) {
		currentCursor = selectedBrushMode === 'brush' ? 'brush' : 'eraser';
	}

	function hideBrushCursor(event: MouseEvent) {
		currentCursor = 'default';
	}

	function paintOnCanvas(x: number, y: number, brushSize: number, canvas: HTMLCanvasElement) {
		// const yScaled = Math.abs(y) * ImgResToCanvasSizeRatio;
		//scale to website pixels from canvas res
		let brushSizeScaled = brushSize * ImgResToCanvasSizeRatio;
		let ctx = canvas.getContext('2d', { willReadFrequently: true });
		// Draw on canvas
		if (ctx) {
			if (prevMouseX === x && prevMouseY === y) {
				ctx.beginPath();
				ctx.arc(x, y, brushSizeScaled / 2, 0, 2 * Math.PI);
				ctx.fillStyle = 'rgba(89, 156, 255, 1)';
				ctx.fill();
			} else {
				ctx.strokeStyle = 'rgba(89, 156, 255, 1)';
				ctx.beginPath();
				ctx.lineJoin = 'round';
				ctx.lineCap = 'round';
				ctx.lineWidth = brushSizeScaled;
				ctx.moveTo(prevMouseX, prevMouseY);
				ctx.lineTo(x, y);
				//color
				ctx.closePath();
				ctx.stroke();
			}

			prevMouseX = x;
			prevMouseY = y;
		}
	}

	function getImageData(canvas: HTMLCanvasElement) {
		return canvas.getContext('2d')!.getImageData(0, 0, canvas.width, canvas.height);
	}

	function createMaskArray(maskCanvas: HTMLCanvasElement) {
		const ctx = maskCanvas.getContext('2d', { willReadFrequently: true });
		if (ctx) {
			const imageData = ctx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
			const maskArray: any = [];
			for (let y = 0; y < maskCanvas.height; y++) {
				const row = [];
				for (let x = 0; x < maskCanvas.width; x++) {
					const index = (y * maskCanvas.width + x) * 4; //RGBA
					const alpha = imageData.data[index + 3]; // Alpha value indicates if the pixel is drawn to
					row.push(alpha > 128); //mark as masked pixels with alpha > 128 (minimizes aliasing better than >0)
				}
				maskArray.push(row);
			}
			return maskArray;
		}
	}

	function handleBrushModeChange(event: any, maskCanvas: HTMLCanvasElement) {
		selectedBrushMode = event.target.value;
		// Get the canvas element
		const context = maskCanvas.getContext('2d');
		if (!context) return;

		context.globalCompositeOperation =
			selectedBrushMode === 'brush' ? 'source-over' : 'destination-out';
	}

	function reshapeBufferToNCHW(
		rgbBuffer: Uint8Array,
		batchSize = 1,
		numChannels: number,
		imageWidth: number,
		imageHeight: number
	) {
		const nchwBuffer = new Uint8Array(batchSize * numChannels * imageWidth * imageHeight);
		for (let i = 0; i < imageWidth * imageHeight; i++) {
			for (let c = 0; c < numChannels; c++) {
				for (let b = 0; b < batchSize; b++) {
					const indexInNCHW =
						b * numChannels * imageWidth * imageHeight + c * imageWidth * imageHeight + i;
					const indexInRGB = i * numChannels + c;
					nchwBuffer[indexInNCHW] = rgbBuffer[indexInRGB];
				}
			}
		}
		return nchwBuffer;
	}

	function booleanMaskToUint8Buffer(maskArray: boolean[][]) {
		const height = maskArray.length;
		const width = maskArray[0].length;

		// Create a new Uint8Array to hold the converted buffer
		const uint8Buffer = new Uint8Array(height * width);

		for (let i = 0; i < height; i++) {
			for (let j = 0; j < width; j++) {
				// Convert boolean value to uint8 (0 or 1)
				uint8Buffer[i * width + j] = maskArray[i][j] ? 0 : 255;
			}
		}

		return uint8Buffer;
	}

	function reshapeCHWtoHWC(
		chwBuffer: Uint8Array,
		width: number,
		height: number,
		channels: number = 3
	) {
		const hwcBuffer = new Uint8Array(width * height * channels);

		for (let c = 0; c < channels; c++) {
			for (let h = 0; h < height; h++) {
				for (let w = 0; w < width; w++) {
					const chwIndex = c * height * width + h * width + w;
					const hwcIndex = h * width * channels + w * channels + c;
					hwcBuffer[hwcIndex] = chwBuffer[chwIndex];
				}
			}
		}

		return hwcBuffer;
	}

	function renderImageToCanvas(
		imageDataRGB: Uint8Array,
		canvas: HTMLCanvasElement,
		img_height: number,
		img_width: number
	) {
		// Create an ImageData object
		const canvasContext = canvas.getContext('2d');
		canvas.width = img_width;
		canvas.height = img_height;
		// Clear the canvas

		let dataRGBBufferReshaped = reshapeCHWtoHWC(imageDataRGB, img_width, img_height);
		let imgDataBuffer = new Uint8ClampedArray(img_height * img_width * 4);

		// fill the imgData buffer, adding alpha channel
		for (let i = 0; i < img_height * img_width; i++) {
			imgDataBuffer[i * 4] = dataRGBBufferReshaped[i * 3];
			imgDataBuffer[i * 4 + 1] = dataRGBBufferReshaped[i * 3 + 1];
			imgDataBuffer[i * 4 + 2] = dataRGBBufferReshaped[i * 3 + 2];
			imgDataBuffer[i * 4 + 3] = 255;
		}

		// const reshaped = reshapeCHWtoHWC(imageData, img_width, img_height);
		// Draw the ImageData onto the canvas
		const imgdata = new ImageData(imgDataBuffer, img_width, img_height);
		canvasContext?.putImageData(imgdata, 0, 0);
	}

	async function runInpainting() {
		// let imgData = getImageData(imageCanvas);
		let imgUInt8Array = imgDataToRGBArray(imgDataOriginal).rgbArray;

		let nchwBuffer = reshapeBufferToNCHW(
			imgUInt8Array,
			1,
			3,
			imageCanvas.width,
			imageCanvas.height
		);
		let imgNCHWTensor = new Tensor('uint8', nchwBuffer, [
			1,
			3,
			imageCanvas.height,
			imageCanvas.width
		]);
		const maskArray = masksStatesHistoryStack[masksStatesHistoryStack.length - 1];
		let maskUInt8Buffer = booleanMaskToUint8Buffer(maskArray);
		let maskNCHWTensor = new Tensor('uint8', maskUInt8Buffer, [
			1,
			1,
			imageCanvas.height,
			imageCanvas.width
		]);

		const output = await onnxSessionMIGAN?.run({ image: imgNCHWTensor, mask: maskNCHWTensor });

		if (output) {
			inpaintedImgCanvas.width = imageCanvas.width;
			inpaintedImgCanvas.height = imageCanvas.height;
			// renderImageToCanvas(output['result'].data, inpaintedImgCanvas, imageCanvas.height, imageCanvas.width);
			let result: Uint8Array = output['result'].data as Uint8Array;
			renderImageToCanvas(result, inpaintedImgCanvas, imageCanvas.height, imageCanvas.width);
		}
	}

	function drawMask(maskCanvas: HTMLCanvasElement, maskArray: any) {
		const maskCanvasctx = maskCanvas.getContext('2d');
		if (maskCanvasctx) {
			const prevMode = maskCanvasctx.globalCompositeOperation;
			maskCanvasctx.globalCompositeOperation = 'source-over';
			clearCanvas(maskCanvas);
			maskCanvasctx.fillStyle = 'rgba(89, 156, 255, 1)';

			for (let y = 0; y < maskArray.length; y++) {
				for (let x = 0; x < maskArray[y].length; x++) {
					if (maskArray[y][x]) {
						maskCanvasctx.fillRect(x, y, 1, 1);
					}
				}
			}
			maskCanvasctx.globalCompositeOperation = prevMode;
		}
	}

	function clearCanvas(canvas: HTMLCanvasElement) {
		const ctx = canvas.getContext('2d');
		if (ctx) {
			ctx.clearRect(0, 0, canvas.width, canvas.height);
		}
	}

	//ONMOUNT MODELS LOADINGS FUNCTIONS
	async function loadOnnxModel() {
		try {
			onnxSession = await InferenceSession.create(mobileSAMEncoderPath, {
				executionProviders: ['wasm'],
				graphOptimizationLevel: 'all'
			});
			onnxSessionMIGAN = await InferenceSession.create(mobile_inpainting_GAN, {
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
		console.log('loaded');
		isLoading = false;
	});
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
	<div id="inpaintBtn">
		<button on:click={runInpainting}>Inpaint</button>
	</div>
	<canvas id="inpaintedImageTmp" bind:this={inpaintedImgCanvas} />

	<button on:click={runModelDecoder} disabled={isEmbedderRunning}>Run Decoder</button>
	<button on:click={undoLastAction} disabled={masksStatesHistoryStack.length === 0}>Undo</button>
	<button on:click={redoLastAction} disabled={masksStatesUndoedStack.length === 0}>Redo</button>
	<!-- range slider to set brush size -->
	<label for="brushSize">Brush size: {brushSize}</label>
	<input type="range" min="1" max="500" bind:value={brushSize} />
	<!-- <button on:click={() => switchToBrushDrawingMode(maskCanvas)}>Brush</button>
	<button on:click={() => switchToBrushErasingMode(maskCanvas)}>Eraser</button> -->

	<div>
		<label>
			<input type="radio" bind:group={selectedTool} value="segment_anything" />
			AI object selector
		</label>
		<label>
			<input type="radio" bind:group={selectedTool} value="brush" />
			Brush
		</label>
	</div>

	{#if selectedTool === 'brush'}
		<div>
			<label>
				<input
					type="radio"
					bind:group={selectedBrushMode}
					value="brush"
					on:change={(e) => handleBrushModeChange(e, maskCanvas)}
				/>
				Brush
			</label>
			<label>
				<input
					type="radio"
					bind:group={selectedBrushMode}
					value="eraser"
					on:change={(e) => handleBrushModeChange(e, maskCanvas)}
				/>
				Eraser
			</label>
		</div>
	{/if}

	<div
		class="canvases"
		on:mouseenter={showBrushCursor}
		on:mouseleave={hideBrushCursor}
		style="cursor: {selectedTool === 'segment_anything'
			? 'default'
			: currentCursor === 'default'
			? 'auto'
			: 'none'}"
		role="group"
	>
		<div
			id="brushToolCursor"
			style="
			display: {selectedTool === 'segment_anything'
				? 'none'
				: currentCursor === 'default'
				? 'none'
				: 'block'};
			width: {brushSize}px;
			height: {brushSize}px;
			left: {currentCanvasRelativeX}px;
			top: {currentCanvasRelativeY}px;
			background-color: {selectedBrushMode === 'brush' ? '#599cff' : '#f5f5f5'};
			opacity: {isPainting ? 0.5 : 0.3};

		"
		/>
		<canvas id="imageCanvas" bind:this={imageCanvas} />

		<canvas
			id="maskCanvas"
			bind:this={maskCanvas}
			on:mousedown={selectedTool === 'brush' ? startPainting : undefined}
			on:mouseup={selectedTool === 'brush' ? stopPainting : undefined}
			on:mousemove={(event) =>
				selectedTool === 'brush' ? handleEditorMouseMove(event, maskCanvas) : undefined}
			on:click={selectedTool === 'segment_anything' ? handleCanvasClick : undefined}
			on:contextmenu={selectedTool === 'segment_anything' ? handleCanvasClick : undefined}
		/>
	</div>
</div>

<!-- TODO AI tool select -->

<style>
	.canvases {
		width: 60%;
		position: relative;
	}
	.canvases canvas {
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
		display: block;
		box-shadow: 0 0 10px 0 rgba(0, 0, 0, 0.2);
	}
	.canvases #maskCanvas {
		opacity: 0.5;
	}
	#brushToolCursor {
		position: absolute;
		overflow: hidden;
		border-radius: 50%;
		pointer-events: none;
		z-index: 100;
		transform: translate(-50%, -50%);
	}
</style>
