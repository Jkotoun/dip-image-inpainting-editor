<script lang="ts">
	import { onMount } from 'svelte';
	import * as tf from '@tensorflow/tfjs';
	import { uploadedImgBase64, uploadedImgFileName } from '../../stores';
	import { Brush, WandSparkles, Undo, Redo, RotateCw, Eraser, Download, HardDriveDownload, AlignCenter } from 'lucide-svelte';
	// import { Tensor } from 'onnxruntime-web';

	import * as ort from 'onnxruntime-web';
	import { AppShell, Tab, TabGroup } from '@skeletonlabs/skeleton';
	// import * as ort from 'onnxruntime-web/webgpu';
	//models paths
	const mobileSAMEncoderPath = '/mobile_sam.encoder.onnx';
	const mobileSAMDecoderPath = '/tfjs_decoder_mobile/model.json';
	// const mobileSAMDecoderPath = '/tfjs_tiny_decoder_quantized/model.json';

	const mobile_inpainting_GAN = '/migan_pipeline_v2.onnx';

	let uploadedImage: string | null = null;

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
	let decoderLoading = false;
	//segmentation model constants
	const longSideLength = 1024;
	let model: tf.GraphModel<string | tf.io.IOHandler>;
	let onnxSession: ort.InferenceSession | null = null;
	let onnxSessionMIGAN: ort.InferenceSession | null = null;
	let resizedImgWidth: number;
	let resizedImgHeight: number;
	// let loadedImgRGB: loadedImgRGBData;

	//inpaint model constants
	let inpaintedImgCanvas: any;

	//editor globals - image and canvas interactions
	let imagebase64URL: any;
	// let uploadedImage: HTMLImageElement | null = null;
	let ImgResToCanvasSizeRatio: number = 1;
	let imageCanvas: any;
	let canvasesContainer: any;
	let maskCanvas: any;
	let isPainting = false;
	let originalImgElement: HTMLImageElement;
	let pixelsDilatation = 10;

	//OLD STATE MANAGEMENT
	interface SAMmarker {
		x: number;
		y: number;
		type: pointType;
	}

	// let clickedPositions: SAMmarker[] = [];

	//editor state management
	interface editorState {
		maskBrush: boolean[][];
		maskSAM: boolean[][];
		maskSAMDilated: boolean[][];
		clickedPositions: SAMmarker[];
		imgData: ImageData;
		currentImgEmbedding: tf.Tensor<tf.Rank> | undefined;
	}
	let imgDataOriginal: ImageData;
	let imgName: string = 'default';
	let currentEditorState: editorState;
	let editorStatesHistory: editorState[] = [];
	//saved for potential redos after undo actions, emptied on new action after series of undos
	let editorStatesUndoed: editorState[] = [];

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

	function changeDilatation(pixelsDilatation: number) {
		if (currentEditorState) {
			currentEditorState.maskSAMDilated = dilateMaskByPixels(
				pixelsDilatation,
				currentEditorState.maskSAM
			);
			renderEditorState(currentEditorState, imageCanvas, maskCanvas);
		}
	}

	$: changeDilatation(pixelsDilatation);

	//EMBEDDING FUNCTIONS
	async function getResizedImgRGBArray(
		img: ImageData,
		longSideLength: number = 1024
	): Promise<loadedImgRGBData> {
		//longerside
		let imgBitmap: ImageBitmap = await createImageBitmap(img);
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
		tempContext?.drawImage(imgBitmap, 0, 0, newWidth, newHeight);
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

	const setupEditor = async (sourceImgBase64Data: string, sourceImgName: string) => {
		const img = new Image();
		img.src = sourceImgBase64Data;
		imgName = sourceImgName;
		img.onload = async () => {
			//setup canvases

			// Calculate aspect ratio
			imageCanvas.width = img.width;
			maskCanvas.width = img.width;
			imageCanvas.height = img.height;
			maskCanvas.height = img.height;

			if (imageCanvas.width > imageCanvas.height) {
				imageCanvas.style.width = maskCanvas.style.width = originalImgElement.style.width = '100%';
				imageCanvas.style.height =
					maskCanvas.style.height =
					originalImgElement.style.height =
						'auto';
			} else {
				imageCanvas.style.width = maskCanvas.style.width = originalImgElement.style.width = 'auto';
				imageCanvas.style.height =
					maskCanvas.style.height =
					originalImgElement.style.height =
						'70vh';
			}
			const canvasElementSize = imageCanvas.getBoundingClientRect();
			canvasesContainer.style.height = `${canvasElementSize.height}px`;
			ImgResToCanvasSizeRatio = img.width / canvasElementSize.width;
			//render image
			const ctx = imageCanvas.getContext('2d');
			clearCanvas(imageCanvas);
			ctx.drawImage(img, 0, 0, img.width, img.height, 0, 0, imageCanvas.width, imageCanvas.height);

			imgDataOriginal = getImageData(imageCanvas);
			//setup editor state
			editorStatesHistory = [];
			editorStatesUndoed = [];
			currentEditorState = {
				maskBrush: new Array(imgDataOriginal.height)
					.fill(false)
					.map(() => new Array(imgDataOriginal.width).fill(false)),
				maskSAM: new Array(imgDataOriginal.height)
					.fill(false)
					.map(() => new Array(imgDataOriginal.width).fill(false)),
				maskSAMDilated: new Array(imgDataOriginal.height)
					.fill(false)
					.map(() => new Array(imgDataOriginal.width).fill(false)),
				clickedPositions: new Array<SAMmarker>(),
				imgData: imgDataOriginal,
				currentImgEmbedding: undefined
			} as editorState;

			isEmbedderRunning = true;
			//0s timeout to handle UI loading state
			setTimeout(async () => {
				currentEditorState.currentImgEmbedding = await runModelEncoder(
					onnxSession!,
					currentEditorState.imgData
				);
				isEmbedderRunning = false;
			}, 0);
		};
	};

	// uploadedImage = img;

	// const handleFileInputChange = async (event: Event) => {
	// 	console.log(event)
	// 	const files = (event.target as HTMLInputElement).files;
	// 	console.log(files)
	// 	let uploadedImageFile: File = files?.[0] as File;
	// 	if (!uploadedImageFile) return;
	// 	imgName = uploadedImageFile.name.substring(0, uploadedImageFile.name.lastIndexOf('.'));
	// 	//clean clicked positions
	// 	// clickedPositions = [];
	// 	//new states management
	// 	editorStatesHistory = [];
	// 	let reader = new FileReader();
	// 	reader.onload = async (e) => {
	// 		imagebase64URL = e.target?.result as string;
	// 		const img = new Image();
	// 		img.src = imagebase64URL as string;
	// 		img.onload = async () => {
	// 			// Calculate aspect ratio
	// 			imageCanvas.width = img.width;
	// 			maskCanvas.width = img.width;
	// 			imageCanvas.height = img.height;
	// 			maskCanvas.height = img.height;
	// 			// initializeMask(imageCanvas.width, imageCanvas.height);

	// 			if (imageCanvas.width > imageCanvas.height) {
	// 				imageCanvas.style.width = maskCanvas.style.width = originalImgElement.style.width = '100%';
	// 				imageCanvas.style.height = maskCanvas.style.height = originalImgElement.style.height ='auto';
	// 			} else {
	// 				imageCanvas.style.width = maskCanvas.style.width = originalImgElement.style.width ='auto';
	// 				imageCanvas.style.height = maskCanvas.style.height = originalImgElement.style.height ='70vh';
	// 			}
	// 			const canvasElementSize = imageCanvas.getBoundingClientRect();
	// 			ImgResToCanvasSizeRatio = img.width / canvasElementSize.width;

	// 			const ctx = imageCanvas.getContext('2d');
	// 			if (uploadedImage) {
	// 				clearCanvas(imageCanvas);
	// 				ctx.drawImage(
	// 					uploadedImage,
	// 					0,
	// 					0,
	// 					uploadedImage.width,
	// 					uploadedImage.height,
	// 					0,
	// 					0,
	// 					imageCanvas.width,
	// 					imageCanvas.height
	// 				);
	// 				//set initial state
	// 				imgDataOriginal = getImageData(imageCanvas);
	// 				currentEditorState = {
	// 					maskBrush: new Array(imgDataOriginal.height)
	// 						.fill(false)
	// 						.map(() => new Array(imgDataOriginal.width).fill(false)),
	// 					maskSAM: new Array(imgDataOriginal.height)
	// 						.fill(false)
	// 						.map(() => new Array(imgDataOriginal.width).fill(false)),
	// 					maskSAMDilated: new Array(imgDataOriginal.height)
	// 						.fill(false)
	// 						.map(() => new Array(imgDataOriginal.width).fill(false)),
	// 					clickedPositions: new Array<SAMmarker>(),
	// 					imgData: imgDataOriginal,
	// 					currentImgEmbedding: undefined
	// 				} as editorState;
	// 				editorStatesHistory = [];
	// 				editorStatesUndoed = [];
	// 			}

	// 			// drawImage(imageCanvas);
	// 			// drawMarkers(imageCanvas, clickedPositions);

	// 			//input for SAM encoder
	// 			if (imgDataOriginal && onnxSession) {
	// 				// isEmbedderRunning = true;
	// 				const embedder_output = await runModelEncoder(onnxSession, imgDataOriginal);

	// 				if (embedder_output) {
	// 					currentEditorState.currentImgEmbedding = embedder_output;
	// 					isEmbedderRunning = false;
	// 				}
	// 			}
	// 		};
	// 		uploadedImage = img;
	// 	};

	// 	isEmbedderRunning = true;
	// 	reader.readAsDataURL(uploadedImageFile);
	// };

	async function runModelEncoder(
		embedderOnnxSession: ort.InferenceSession,
		imageData: ImageData
		// imgRGBData: loadedImgRGBData
	): Promise<tf.Tensor<tf.Rank>> {
		let resizedImgRGBData = await getResizedImgRGBArray(imageData, longSideLength);
		// const input = new Float32Array(imgTensor.dataSync());
		let floatArray = Float32Array.from(resizedImgRGBData.rgbArray);
		const inputTensor = new ort.Tensor('float32', floatArray, [
			resizedImgRGBData.height,
			resizedImgRGBData.width,
			3
		]);
		const output = await embedderOnnxSession.run({ input_image: inputTensor });
		return tf.tensor(
			output['image_embeddings'].data as any,
			output['image_embeddings'].dims as any,
			'float32'
		);
	}

	//DECODER MODEL FUNCTIONS

	async function createInputDict(currentEditorState: editorState) {
		let inputPoint: Array<{ x: number; y: number }> = currentEditorState.clickedPositions.map(
			({ x, y }) => coordsToResizedImgScale(x, y)
		);
		let inputLabels: Array<number> = currentEditorState.clickedPositions.map(({ type }) =>
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
			image_embeddings: currentEditorState.currentImgEmbedding as tf.Tensor<tf.Rank>,
			point_coords: pointCoordsTensor,
			point_labels: pointLabelsTensor,
			mask_input: maskInputTensor,
			has_mask_input: hasMaskInputTensor,
			orig_im_size: origImgSizeTensor
		};
		return modelInput;
	}

	async function runModelDecoder(
		currentEditorState: editorState
	): Promise<boolean[][] | undefined> {
		if (!currentEditorState.currentImgEmbedding) {
			return;
		}
		const modelInput = await createInputDict(currentEditorState);

		// let data = modelInput.image_embeddings.arraySync();
		const predictions: any = await model.executeAsync(modelInput);
		console.log('executed');
		const lastData = await predictions[predictions.length - 1].arraySync();
		const data = lastData[0][0];

		//convert to bool array
		const SAMMaskArray = data.map((val: number[]) => val.map((v) => v > 0.0));
		return SAMMaskArray;
		// return dilateMaskByPixels(pixelsDilatation, SAMMaskArray);

		// return SAMMaskArray;
	}

	//EDITOR HANDLING FUNCTIONS

	function coordsToResizedImgScale(x: number, y: number) {
		const imageX = (x / imageCanvas.width) * resizedImgWidth;
		const imageY = (y / imageCanvas.height) * resizedImgHeight;
		return { x: imageX, y: imageY };
	}

	async function handleCanvasClick(event: MouseEvent) {
		//it logs -0 at 0,0 for some reason
		event.preventDefault();
		const xScaled = Math.abs(event.offsetX) * ImgResToCanvasSizeRatio;
		const yScaled = Math.abs(event.offsetY) * ImgResToCanvasSizeRatio;

		editorStatesHistory = [...editorStatesHistory, currentEditorState];
		currentEditorState = {
			maskBrush: currentEditorState.maskBrush,
			maskSAM: currentEditorState.maskSAM,
			maskSAMDilated: currentEditorState.maskSAMDilated,
			clickedPositions: new Array<SAMmarker>(...currentEditorState.clickedPositions, {
				x: xScaled,
				y: yScaled,
				type: event.button === 0 ? 'positive' : 'negative'
			}),
			imgData: currentEditorState.imgData,
			currentImgEmbedding: currentEditorState.currentImgEmbedding
		} as editorState;
		console.log('here');
		renderEditorState(currentEditorState, imageCanvas, maskCanvas);
		decoderLoading = true;
		setTimeout(() => {
			runModelDecoder(currentEditorState).then((sammask) => {
				if (sammask) {
					currentEditorState.maskSAM = sammask;
					currentEditorState.maskSAMDilated = dilateMaskByPixels(pixelsDilatation, sammask);
				}
				renderEditorState(currentEditorState, imageCanvas, maskCanvas);
				decoderLoading = false;
			});
		}, 0);
	}

	function drawImage(canvas: HTMLCanvasElement, imageData: ImageData) {
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
		//new state management
		if (editorStatesHistory.length > 0 && currentEditorState) {
			editorStatesUndoed = [...editorStatesUndoed, currentEditorState];
			currentEditorState = editorStatesHistory[editorStatesHistory.length - 1];
			editorStatesHistory = editorStatesHistory.slice(0, -1);
			renderEditorState(currentEditorState, imageCanvas, maskCanvas);
		}
	}

	function redoLastAction() {
		//new state management
		if (editorStatesUndoed.length > 0 && currentEditorState) {
			editorStatesHistory = [...editorStatesHistory, currentEditorState];
			currentEditorState = editorStatesUndoed[editorStatesUndoed.length - 1];
			editorStatesUndoed = editorStatesUndoed.slice(0, -1);
			renderEditorState(currentEditorState, imageCanvas, maskCanvas);
		}
	}

	function reset() {
		//clear all states
		currentEditorState = {
			maskBrush: new Array(imgDataOriginal.height)
				.fill(false)
				.map(() => new Array(imgDataOriginal.width).fill(false)),
			maskSAM: new Array(imgDataOriginal.height)
				.fill(false)
				.map(() => new Array(imgDataOriginal.width).fill(false)),
			maskSAMDilated: new Array(imgDataOriginal.height)
				.fill(false)
				.map(() => new Array(imgDataOriginal.width).fill(false)),
			clickedPositions: [],
			imgData: imgDataOriginal,
			currentImgEmbedding: editorStatesHistory[0].currentImgEmbedding
		};
		editorStatesHistory = [];
		editorStatesUndoed = [];
		renderEditorState(currentEditorState, imageCanvas, maskCanvas);
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
		let maskArray: boolean[][] = createMaskArray(maskCanvas);
		//new state management
		editorStatesHistory = [...editorStatesHistory, currentEditorState];
		currentEditorState = {
			maskBrush: maskArray,
			maskSAM: currentEditorState.maskSAM,
			maskSAMDilated: currentEditorState.maskSAMDilated,
			clickedPositions: currentEditorState.clickedPositions,
			imgData: currentEditorState.imgData,
			currentImgEmbedding: currentEditorState.currentImgEmbedding
		} as editorState;
		editorStatesUndoed = [];
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

	function createMaskArray(maskCanvas: HTMLCanvasElement): boolean[][] {
		const ctx = maskCanvas.getContext('2d', { willReadFrequently: true });
		if (ctx) {
			const imageData = ctx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);

			// let maskArray: boolean[][] = [];
			// maskArray.map((row: boolean[], y: number) => {
			// 	row.map((val, x) => {
			// 		const index = (y * maskCanvas.width + x) * 4; //RGBA
			// 		const alpha = imageData.data[index + 3]; // Alpha value indicates if the pixel is drawn to
			// 		row.push(alpha > 128); //mark as masked pixels with alpha > 128 (minimizes aliasing better than >0)
			// 	});
			// });

			const maskArray: boolean[][] = [];
			for (let y = 0; y < maskCanvas.height; y++) {
				const row: boolean[] = [];
				for (let x = 0; x < maskCanvas.width; x++) {
					const index = (y * maskCanvas.width + x) * 4; //RGBA
					const alpha = imageData.data[index + 3]; // Alpha value indicates if the pixel is drawn to
					row.push(alpha > 128); //mark as masked pixels with alpha > 128 (minimizes aliasing better than >0)
				}
				maskArray.push(row);
			}
			return maskArray;
		}
		return [];
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

	function RGB_CHW_array_to_imageData(
		imageDataRGB: Uint8Array,
		img_height: number,
		img_width: number
	) {
		// Create an ImageData object
		// const canvasContext = canvas.getContext('2d');
		// canvas.width = img_width;
		// canvas.height = img_height;
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
		return new ImageData(imgDataBuffer, img_width, img_height);
	}

	async function runInpainting(currentEditorState: editorState): Promise<ImageData> {
		let imgUInt8Array = imgDataToRGBArray(currentEditorState.imgData).rgbArray;

		let nchwBuffer = reshapeBufferToNCHW(
			imgUInt8Array,
			1,
			3,
			imageCanvas.width,
			imageCanvas.height
		);
		let imgNCHWTensor = new ort.Tensor('uint8', nchwBuffer, [
			1,
			3,
			imageCanvas.height,
			imageCanvas.width
		]);

		//combine currentstate brushmask and sammask with or
		let maskArrayCombined = currentEditorState.maskBrush.map((row: boolean[], y: number) =>
			row.map((val, x) => val || currentEditorState.maskSAMDilated[y][x])
		);

		let maskUInt8Buffer = booleanMaskToUint8Buffer(maskArrayCombined);
		let maskNCHWTensor = new ort.Tensor('uint8', maskUInt8Buffer, [
			1,
			1,
			imageCanvas.height,
			imageCanvas.width
		]);

		const output = await onnxSessionMIGAN?.run({ image: imgNCHWTensor, mask: maskNCHWTensor });

		let result: Uint8Array = output!['result'].data as Uint8Array;
		let resultImgData = RGB_CHW_array_to_imageData(result, imageCanvas.height, imageCanvas.width);
		return resultImgData;
	}

	function renderEditorState(
		state: editorState,
		imageCanvas: HTMLCanvasElement,
		maskCanvas: HTMLCanvasElement
	) {
		clearCanvas(imageCanvas);
		clearCanvas(maskCanvas);
		drawImage(imageCanvas, state.imgData);
		drawMarkers(imageCanvas, state.clickedPositions);
		drawMask(imageCanvas, state.maskSAMDilated, 0.5, false);
		drawMask(maskCanvas, state.maskBrush, 1, true);
	}

	// function drawMask(maskCanvas: HTMLCanvasElement, maskArray: boolean[][]) {
	// 	const maskCanvasctx = maskCanvas.getContext('2d');
	// 	if (maskCanvasctx) {
	// 		// //map bool array to imageData
	// 		let imageData: ImageData = maskCanvasctx.createImageData(maskArray[0].length, maskArray.length);
	// 		maskArray.map((row: boolean[], y: number) => {
	// 			row.map((val, x) => {
	// 				let index = (y * maskArray[0].length + x) * 4;
	// 				if(val){
	// 					imageData.data[index] = 89;
	// 					imageData.data[index + 1] = 156;
	// 					imageData.data[index + 2] = 255;
	// 					imageData.data[index + 3] = 128;
	// 				}else{
	// 					imageData.data[index] = 0;
	// 					imageData.data[index + 1] = 0;
	// 					imageData.data[index + 2] = 0;
	// 					imageData.data[index + 3] = 0;
	// 				}
	// 			});
	// 		});
	// 		maskCanvasctx.putImageData(imageData, 0, 0);
	// 	}
	// }
	function drawMask(
		maskCanvas: HTMLCanvasElement,
		maskArray: any,
		opacity: number,
		clearCanvasFirst = false
	) {
		const maskCanvasctx = maskCanvas.getContext('2d');
		if (maskCanvasctx) {
			if (clearCanvasFirst) {
				clearCanvas(maskCanvas);
			}
			const prevMode = maskCanvasctx.globalCompositeOperation;
			maskCanvasctx.globalCompositeOperation = 'source-over';
			maskCanvasctx.fillStyle = `rgba(89, 156, 255, ${opacity})`;

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

	function setInpaintedImgEditorState(inpaintedImgData: ImageData) {
		editorStatesHistory = [...editorStatesHistory, currentEditorState];
		let newEditorState: editorState = {
			maskBrush: new Array(imageCanvas.height)
				.fill(false)
				.map(() => new Array(imageCanvas.width).fill(false)),
			maskSAM: new Array(imageCanvas.height)
				.fill(false)
				.map(() => new Array(imageCanvas.width).fill(false)),
			maskSAMDilated: new Array(imageCanvas.height)
				.fill(false)
				.map(() => new Array(imageCanvas.width).fill(false)),
			clickedPositions: [],
			imgData: inpaintedImgData,
			currentImgEmbedding: undefined
		};
		renderEditorState(newEditorState, imageCanvas, maskCanvas);

		isEmbedderRunning = true;

		setTimeout(() => {
			if (!onnxSession) return;
			runModelEncoder(onnxSession, currentEditorState.imgData).then((newEmbedding) => {
				newEditorState.currentImgEmbedding = newEmbedding;
				currentEditorState = newEditorState;
				isEmbedderRunning = false;
			});
		}, 0);
	}

	function downloadImage(imageData: ImageData) {
		// Create a temporary canvas element
		const tempCanvas = document.createElement('canvas');
		tempCanvas.width = imageData.width;
		tempCanvas.height = imageData.height;
		const ctx = tempCanvas.getContext('2d');

		// Draw the ImageData on the temporary canvas
		ctx?.putImageData(imageData, 0, 0);

		// Convert the temporary canvas to a Blob and download
		tempCanvas.toBlob((blob) => {
			const url = URL.createObjectURL(blob as Blob);
			const a = document.createElement('a');
			a.href = url;
			a.download = `${imgName}_edited.png`; // Name of the file you want to download
			document.body.appendChild(a);
			a.click();
			document.body.removeChild(a);
			URL.revokeObjectURL(url);
		}, 'image/png');
	}

	// function dilateMaskByPixels(pixels: number, mask: boolean[][]): boolean[][] {
	//     let dilatedMask = mask.map(row => row.slice());
	//     for (let p = 0; p < pixels; p++) {
	//         let tempMask = dilatedMask.map(row => row.slice());
	//         for (let i = 0; i < dilatedMask.length; i++) {
	//             for (let j = 0; j < dilatedMask[i].length; j++) {
	//                 if (dilatedMask[i][j]) {
	//                     // Applying dilation (considering direct neighbors for simplicity)
	//                     [[-1, 0], [1, 0], [0, -1], [0, 1]].forEach(([dx, dy]) => {
	//                         let x = i + dx, y = j + dy;
	//                         if (x >= 0 && x < dilatedMask.length && y >= 0 && y < dilatedMask[i].length) {
	//                             tempMask[x][y] = true;
	//                         }
	//                     });
	//                 }
	//             }
	//         }
	//         dilatedMask = tempMask;
	//     }
	//     return dilatedMask;
	// }
	type Point = { x: number; y: number };

	function dilateMaskByPixels(dilationPixels: number, mask: boolean[][]): boolean[][] {
		const numRows = mask.length;
		const numCols = mask[0].length;
		const directions = [
			[-1, 0],
			[1, 0],
			[0, -1],
			[0, 1]
		]; // 4-way connectivity
		let queue: Point[] = [];
		let dilatedMask = mask.map((row) => row.slice());

		// Initialize the queue with the edge pixels
		for (let i = 0; i < numRows; i++) {
			for (let j = 0; j < numCols; j++) {
				if (mask[i][j]) {
					directions.forEach(([dx, dy]) => {
						const newX = i + dx;
						const newY = j + dy;
						if (
							newX >= 0 &&
							newX < numRows &&
							newY >= 0 &&
							newY < numCols &&
							!dilatedMask[newX][newY]
						) {
							dilatedMask[newX][newY] = true;
							queue.push({ x: newX, y: newY });
						}
					});
				}
			}
		}

		// Perform dilation for the specified number of pixels
		for (let p = 0; p < dilationPixels - 1; p++) {
			// -1 because we already did one dilation
			let newQueue: Point[] = [];
			while (queue.length > 0) {
				const { x, y } = queue.shift()!;
				directions.forEach(([dx, dy]) => {
					const newX = x + dx;
					const newY = y + dy;
					if (
						newX >= 0 &&
						newX < numRows &&
						newY >= 0 &&
						newY < numCols &&
						!dilatedMask[newX][newY]
					) {
						dilatedMask[newX][newY] = true;
						newQueue.push({ x: newX, y: newY });
					}
				});
			}
			queue = newQueue;
		}

		return dilatedMask;
	}

	//ONMOUNT MODELS LOADINGS FUNCTIONS
	async function loadOnnxModels() {
		try {
			onnxSession = await ort.InferenceSession.create(mobileSAMEncoderPath, {
				executionProviders: ['wasm'],
				graphOptimizationLevel: 'all'
			});
			onnxSessionMIGAN = await ort.InferenceSession.create(mobile_inpainting_GAN, {
				executionProviders: ['wasm'],
				graphOptimizationLevel: 'all'
			});
		} catch (error) {
			console.error('Error loading the ONNX model:', error);
		}
	}

	onMount(async () => {
		window.addEventListener('resize', () => {
			if (imageCanvas) {
				const canvasElementSize = imageCanvas.getBoundingClientRect();
				ImgResToCanvasSizeRatio = imageCanvas.width / canvasElementSize.width;
			}
		});
		await tf.ready();
		try {
			await import('@tensorflow/tfjs-backend-webgpu');
			await tf.setBackend('webgpu');
			console.log('webgpu loaded');
		} catch (e) {
			try {
				await tf.setBackend('webgl');
			} catch (e) {
				console.error('could not load backend:', e);
				throw e;
			}
		}
		await loadOnnxModels();
		model = await tf.loadGraphModel(mobileSAMDecoderPath);
		uploadedImage = $uploadedImgBase64;

		if (!uploadedImage) {
			return;
		}
		if (!onnxSession || !onnxSessionMIGAN) {
			console.error('models not loaded');
			return;
		} else {
			console.log('model loaded');
		}
		isLoading = false;
		setupEditor(uploadedImage, $uploadedImgFileName);
	});
</script>

<AppShell>
	<div slot="sidebarLeft">
		<TabGroup>
			<Tab class="px-8 py-4" bind:group={selectedTool} name="segment_anything" value="segment_anything">
				<span class="flex gap-x-2 items-center"> <WandSparkles size={18} /> Smart selector</span>
			</Tab>
			<Tab class="px-8 py-4" bind:group={selectedTool} name="brush" value="brush">
				<span class="flex gap-x-2 items-center"> <Brush size={18} /> Brush</span>
			</Tab>

			<div slot="panel" class="p-4">
				{#if selectedTool === 'brush'}
					<div>
						<label>
							<input
								type="radio"
								bind:group={selectedBrushMode}
								value="brush"
								on:change={(e) => handleBrushModeChange(e, maskCanvas)}
							/>
							Brush <Brush/>
						</label>
						<label>
							<input
								type="radio"
								bind:group={selectedBrushMode}
								value="eraser"
								on:change={(e) => handleBrushModeChange(e, maskCanvas)}
							/>
							Eraser <Eraser/>
						</label>

						<label for="brushSize">Brush size: {brushSize}</label>
						<input type="range" min="1" max="500" bind:value={brushSize} />
					</div>
				{:else if selectedTool === 'segment_anything'}
					<label for="pixelsDilatation">Dilatation: {pixelsDilatation}</label>
					<input type="range" min="0" max="25" bind:value={pixelsDilatation} />
				{/if}
			</div>
		</TabGroup>
	</div>
	<div class="px-64 py-8">
		{#if isLoading}
			<h3>Loading model...</h3>
		{:else if isEmbedderRunning}
			<h3>Running embedder...</h3>
		{/if}
		<!-- top buttons panel -->
		<div class="flex py-2 justify-between ">
			<!-- left buttons -->
			<div class="flex gap-x-2">
				<button
					class="btn variant-filled"
					on:click={undoLastAction}
					disabled={editorStatesHistory.length === 0}><Undo/></button
				>
				<button
					class="btn variant-filled"
					on:click={redoLastAction}
					disabled={editorStatesUndoed.length === 0}><Redo/></button
				>
				<button class="btn variant-filled" on:click={reset}><RotateCw/></button>
			</div>
			<!-- right buttons -->
			<div class="flex gap-x-2">
				<div
					role="button"
					tabindex="0"
					class="btn variant-filled"
					on:mousedown={() => {
						maskCanvas.style.display = 'none';
						imageCanvas.style.display = 'none';
						originalImgElement.style.display = 'block';
					}}
					on:mouseup={() => {
						maskCanvas.style.display = 'block';
						imageCanvas.style.display = 'block';
						originalImgElement.style.display = 'none';
					}}
				>
					Hold to compare
				</div>
				<button
					class="btn variant-filled"
					on:click={() => downloadImage(currentEditorState.imgData)}> <span class="flex gap-x-2 items-center"><HardDriveDownload size={18}/> Download </span> </button
				>
			</div>
		</div>
		<!-- editor canvases-->
		<div
			class="canvases"
			bind:this={canvasesContainer}
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
				on:click={selectedTool === 'segment_anything'
					? async (e) => handleCanvasClick(e)
					: undefined}
				on:contextmenu={selectedTool === 'segment_anything' ? handleCanvasClick : undefined}
			/>
			<img src={$uploadedImgBase64} alt="originalImage" bind:this={originalImgElement} />
		</div>
		<!-- bottom buttons -->
		<div class="flex justify-end py-2">
			<!-- todo move/zoom control panel -->
			<div></div>
			<button
				class="btn variant-filled"
				on:click={async () => {
					let resultImgData = await runInpainting(currentEditorState);
					setInpaintedImgEditorState(resultImgData);
				}}> <span class="flex gap-x-2 items-center"><WandSparkles size={18}/> Remove </span></button>
		</div>
	</div>
</AppShell>

<style>
	.canvases canvas,
	.canvases img {
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
	.canvases {
		position: relative;
	}
	#brushToolCursor {
		position: absolute;
		overflow: hidden;
		border-radius: 50%;
		pointer-events: none;
		z-index: 100;
		transform: translate(-50%, -50%);
	}

	.canvases img {
		display: none;
	}
</style>
