<script lang="ts">
	import { onMount, tick } from 'svelte';
	import { base } from '$app/paths';
	import { uploadedImgBase64, uploadedImgFileName } from '../../stores/imgStore';
	import { mainWorker } from '../../stores/workerStore';
	import { MESSAGE_TYPES } from '../../workers/messageTypes';
	import Panzoom, { type PanzoomObject } from '@panzoom/panzoom';
	import {
		Brush,
		WandSparkles,
		Undo,
		Redo,
		RotateCw,
		Eraser,
		HardDriveDownload,
		PlusCircleIcon,
		MinusCircle,
		MoveIcon,
		ScanEyeIcon,
		PlusIcon,
		MinusIcon
	} from 'lucide-svelte';
	import {
		AppBar,
		AppShell,
		Drawer,
		getDrawerStore,
		LightSwitch,
		ProgressRadial,
		RadioGroup,
		RadioItem,
		Tab,
		TabGroup
	} from '@skeletonlabs/skeleton';
	import { goto } from '$app/navigation';

	let uploadedImage: string | null = null;

	//types definition
	type pointType = 'positive' | 'negative';

	//interface interaction globals
	interface loadedImgRGBData {
		rgbArray: Uint8Array;
		width: number;
		height: number;
	}

	let encoderLoading = true;
	let decoderLoading = true;
	let inpainterLoading = true;
	let isEmbedderRunning = false;
	let decoderRunning = false;
	let inpaintingRunning = false;
	$: anythingEssentialLoading =
		inpaintingRunning || encoderLoading || decoderLoading || isEmbedderRunning || decoderRunning;

	let enablePan: boolean = false;
	//segmentation model constants
	const longSideLength = 1024;
	let resizedImgWidth: number;
	let resizedImgHeight: number;

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

	//editor state management
	interface editorState {
		maskBrush: boolean[][];
		maskSAM: boolean[][];
		maskSAMDilated: boolean[][];
		clickedPositions: SAMmarker[];
		imgData: ImageData;
		currentImgEmbedding:
			| {
					data: any;
					dims: any;
			  }
			| undefined;
	}
	let imgDataOriginal: ImageData;
	let imgName: string = 'default';
	let currentEditorState: editorState;
	let editorStatesHistory: editorState[] = [];
	//saved for potential redos after undo actions, emptied on new action after series of undos
	let editorStatesUndoed: editorState[] = [];
	let appbarElement: HTMLElement;
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
	let selectedSAMMode: 'positive' | 'negative' = 'positive';
	$: changeDilatation(pixelsDilatation);

	function handleWorkerModelsMessages(event: MessageEvent<any>) {
		const { data, type } = event.data;
		// if(type === MESSAGE_TYPES)
		if (type === MESSAGE_TYPES.INPAINTER_LOADED) {
			inpainterLoading = false;
		}
		if (type === MESSAGE_TYPES.DECODER_RUN_RESULT_SUCCESS && !decoderLoading) {
			const SAMMaskArray = data.map((val: number[]) => val.map((v) => v > 0.0));
			if (SAMMaskArray) {
				dilateMaskByPixels(pixelsDilatation, SAMMaskArray).then((dilatedMask) => {
					currentEditorState.maskSAMDilated = dilatedMask;
					currentEditorState.maskSAM = SAMMaskArray;
					renderEditorState(currentEditorState, imageCanvas, maskCanvas);
				});
			}
			decoderRunning = false;
		} else if (type === MESSAGE_TYPES.ENCODER_RUN_RESULT_SUCCESS && !encoderLoading) {
			// let img_tensor = tf.tensor(data.embeddings as any, data.dims as any, 'float32');
			currentEditorState.currentImgEmbedding = {
				data: data.embeddings,
				dims: data.dims
			};
			isEmbedderRunning = false;
		} else if (type === MESSAGE_TYPES.INPAINTING_RUN_RESULT_SUCCESS && !inpainterLoading) {
			RGB_CHW_array_to_imageData(data, imageCanvas.height, imageCanvas.width).then(
				(resultImgData) => {
					setInpaintedImgEditorState(resultImgData);
					inpaintingRunning = false;
				}
			);
		} else if (type === MESSAGE_TYPES.ALL_MODELS_LOADED) {
			decoderLoading = encoderLoading = inpainterLoading = false;
			//run only if editor state is already set and embedding hasnt been run already
			if (
				currentEditorState &&
				currentEditorState.imgData &&
				!isEmbedderRunning &&
				!currentEditorState.currentImgEmbedding
			) {
				isEmbedderRunning = true;
				runModelEncoder(currentEditorState.imgData);
			}
			// Continue with your logic here...
		} else if (type === MESSAGE_TYPES.ENCODER_DECODER_LOADED) {
			encoderLoading = decoderLoading = false;
			//run only if editor state is already set and embedding hasnt been run already
			if (
				currentEditorState &&
				currentEditorState.imgData &&
				!isEmbedderRunning &&
				!currentEditorState.currentImgEmbedding
			) {
				isEmbedderRunning = true;
				runModelEncoder(currentEditorState.imgData).then((_) => {
					$mainWorker?.postMessage({ type: MESSAGE_TYPES.LOAD_INPAINTER });
				});
			}
		} else if (type === MESSAGE_TYPES.NONE_LOADED) {
		}
	}
	if ($mainWorker) {
		$mainWorker.onmessage = handleWorkerModelsMessages;
	}
	async function changeDilatation(pixelsDilatation: number) {
		if (currentEditorState) {
			currentEditorState.maskSAMDilated = await dilateMaskByPixels(
				pixelsDilatation,
				currentEditorState.maskSAM
			);
			renderEditorState(currentEditorState, imageCanvas, maskCanvas);
		}
	}

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
		return new Promise<void>((resolve, reject) => {
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

				// if (imageCanvas.width > imageCanvas.height) {
				// 	imageCanvas.style.width =
				// 		maskCanvas.style.width =
				// 		originalImgElement.style.width =
				// 			'100%';
				// 	// imageCanvas.style.maxHeight =
				// 	// 	maskCanvas.style.maxHeight =
				// 	// 	originalImgElement.style.maxHeight =
				// 	// 		'75vh';
				// } else {
				// 	imageCanvas.style.width =
				// 		maskCanvas.style.width =
				// 		originalImgElement.style.width =
				// 			'auto';
				// 	imageCanvas.style.height =
				// 		maskCanvas.style.height =
				// 		originalImgElement.style.height =
				// 			'75vh';
				// }
				const canvasElementSize = imageCanvas.getBoundingClientRect();
				// canvasesContainer.style.height = `${canvasElementSize.height}px`;
				ImgResToCanvasSizeRatio = img.width / canvasElementSize.width;
				//render image
				const ctx = imageCanvas.getContext('2d');
				clearCanvas(imageCanvas);
				ctx.drawImage(
					img,
					0,
					0,
					img.width,
					img.height,
					0,
					0,
					imageCanvas.width,
					imageCanvas.height
				);

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
				resolve();

				//0s timeout to handle UI loading state
			};

			img.onerror = (error) => {
				// Reject the Promise if there's an error loading the image
				reject(error);
			};
		});
	};

	async function runModelEncoder(imageData: ImageData): Promise<void> {
		let resizedImgRGBData = await getResizedImgRGBArray(imageData, longSideLength);
		let floatArray = Float32Array.from(resizedImgRGBData.rgbArray);
		$mainWorker?.postMessage({
			type: MESSAGE_TYPES.ENCODER_RUN,
			data: {
				img_array_data: floatArray,
				dims: [resizedImgRGBData.height, resizedImgRGBData.width, 3]
			}
		});
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

		let modelInput = {
			image_embeddings: {
				data: currentEditorState.currentImgEmbedding!.data,
				dims: currentEditorState.currentImgEmbedding!.dims
			},
			point_coords: {
				data: onnxInputPoints.flat(),
				dims: [1, onnxInputPoints.length, 2]
			},
			point_labels: {
				data: inputLabels,
				dims: [1, inputLabels.length]
			},
			mask_input: {
				data: new Float32Array(256 * 256).fill(0),
				dims: [1, 1, 256, 256]
			},
			has_mask_input: {
				data: [0],
				dims: [1]
			},
			orig_im_size: {
				data: [imageCanvas.height, imageCanvas.width],
				dims: [2]
			}
		};
		return modelInput;
	}

	async function runModelDecoder(
		currentEditorState: editorState
	): Promise<boolean[][] | undefined> {
		if (!currentEditorState.currentImgEmbedding) {
			return;
		}
		decoderRunning = true;
		const modelInput = await createInputDict(currentEditorState);
		$mainWorker?.postMessage({
			type: MESSAGE_TYPES.DECODER_RUN,
			data: modelInput
		});
	}

	//EDITOR HANDLING FUNCTIONS

	function coordsToResizedImgScale(x: number, y: number) {
		const imageX = (x / imageCanvas.width) * resizedImgWidth;
		const imageY = (y / imageCanvas.height) * resizedImgHeight;
		return { x: imageX, y: imageY };
	}

	async function handleCanvasClick(event: MouseEvent) {
		if (anythingEssentialLoading) {
			return;
		}
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
				// type: event.button === 0 ? 'positive' : 'negative'
				type: selectedSAMMode
			}),
			imgData: currentEditorState.imgData,
			currentImgEmbedding: currentEditorState.currentImgEmbedding
		} as editorState;
		renderEditorState(currentEditorState, imageCanvas, maskCanvas);
		runModelDecoder(currentEditorState);
	}

	function drawImage(canvas: HTMLCanvasElement, imageData: ImageData) {
		canvas.getContext('2d')!.putImageData(imageData, 0, 0);
	}
	function drawMarkers(canvas: HTMLCanvasElement, clickedPositions: SAMmarker[]) {
		const canvasContext = canvas.getContext('2d');
		if (!canvasContext) return;
		for (const pos of clickedPositions) {
			canvasContext.fillStyle = pos.type === 'positive' ? '#021ded' : 'red';
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
		panzoom.reset();
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
			currentImgEmbedding: editorStatesHistory[0]
				? editorStatesHistory[0].currentImgEmbedding
				: currentEditorState.currentImgEmbedding
		};
		editorStatesHistory = [];
		editorStatesUndoed = [];
		renderEditorState(currentEditorState, imageCanvas, maskCanvas);
	}

	// brush tool
	function startPaintingMouse(event: MouseEvent, canvas: HTMLCanvasElement) {
		isPainting = true;
		prevMouseX = event.offsetX * ImgResToCanvasSizeRatio;
		prevMouseY = event.offsetY * ImgResToCanvasSizeRatio;
		handleEditorMouseMove(event, canvas);
	}
	function startPaintingTouch(event: TouchEvent, canvas: HTMLCanvasElement) {
		isPainting = true;

		let touch = event.touches[0]; // Get the first touch, you might handle multi-touch differently
		let targetRect = canvas.getBoundingClientRect(); // Get the target element's position

		// Calculate offsetX and offsetY
		let offsetX = touch.clientX - targetRect.left;
		let offsetY = touch.clientY - targetRect.top;

		prevMouseX = offsetX * ImgResToCanvasSizeRatio;
		prevMouseY = offsetY * ImgResToCanvasSizeRatio;
		// handleEditorTouchMove(event, canvas);
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
		console.log(event);
		const x = event.offsetX * ImgResToCanvasSizeRatio;
		const y = event.offsetY * ImgResToCanvasSizeRatio;

		currentCanvasRelativeX = event.offsetX;
		currentCanvasRelativeY = event.offsetY;
		if (isPainting) {
			paintOnCanvas(x, y, brushSize, canvas);
		}
	}

	function handleEditorTouchMove(event: TouchEvent, canvas: HTMLCanvasElement) {
		let touch = event.touches[0]; // Get the first touch, you might handle multi-touch differently
		let targetRect = canvas.getBoundingClientRect(); // Get the target element's position

		// Calculate offsetX and offsetY
		let offsetX = touch.clientX - targetRect.left;
		let offsetY = touch.clientY - targetRect.top;
		currentCanvasRelativeX = offsetX;
		currentCanvasRelativeY = offsetY;

		if (isPainting) {
			paintOnCanvas(
				offsetX * ImgResToCanvasSizeRatio,
				offsetY * ImgResToCanvasSizeRatio,
				brushSize,
				canvas
			);
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

	async function RGB_CHW_array_to_imageData(
		imageDataRGB: Uint8Array,
		img_height: number,
		img_width: number
	): Promise<ImageData> {
		let dataRGBBufferReshaped = reshapeCHWtoHWC(imageDataRGB, img_width, img_height);
		let imgDataBuffer = new Uint8ClampedArray(img_height * img_width * 4);
		// fill the imgData buffer, adding alpha channel
		for (let i = 0; i < img_height * img_width; i++) {
			imgDataBuffer[i * 4] = dataRGBBufferReshaped[i * 3];
			imgDataBuffer[i * 4 + 1] = dataRGBBufferReshaped[i * 3 + 1];
			imgDataBuffer[i * 4 + 2] = dataRGBBufferReshaped[i * 3 + 2];
			imgDataBuffer[i * 4 + 3] = 255;
		}

		// Draw the ImageData onto the canvas
		return new ImageData(imgDataBuffer, img_width, img_height);
	}

	async function runInpainting(currentEditorState: editorState): Promise<void> {
		let imgUInt8Array = imgDataToRGBArray(currentEditorState.imgData).rgbArray;
		let nchwBuffer = reshapeBufferToNCHW(
			imgUInt8Array,
			1,
			3,
			imageCanvas.width,
			imageCanvas.height
		);
		//combine currentstate brushmask and sammask with or
		let maskArrayCombined = currentEditorState.maskBrush.map((row: boolean[], y: number) =>
			row.map((val, x) => val || currentEditorState.maskSAMDilated[y][x])
		);

		let maskUInt8Buffer = booleanMaskToUint8Buffer(maskArrayCombined);
		$mainWorker?.postMessage({
			type: MESSAGE_TYPES.INPAINTING_RUN,
			data: {
				imageTensorData: {
					data: nchwBuffer,
					dims: [1, 3, imageCanvas.height, imageCanvas.width]
				},

				maskTensorData: {
					data: maskUInt8Buffer,
					dims: [1, 1, imageCanvas.height, imageCanvas.width]
				}
			}
		});
	}

	async function handleInpainting() {
		if (!anythingEssentialLoading) {
			inpaintingRunning = true;
			runInpainting(currentEditorState);
		}
	}

	async function renderEditorState(
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
			maskCanvasctx.fillStyle = `rgba(64, 141, 255, ${opacity})`;

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

	async function setInpaintedImgEditorState(inpaintedImgData: ImageData) {
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
		currentEditorState = newEditorState;

		renderEditorState(currentEditorState, imageCanvas, maskCanvas);

		isEmbedderRunning = true;
		runModelEncoder(currentEditorState.imgData);
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

	async function dilateMaskByPixels(
		dilationPixels: number,
		mask: boolean[][]
	): Promise<boolean[][]> {
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
	let headerHeightPx = 0;
	onMount(async () => {
		let header = document.querySelector('header');
		if (header) {
			headerHeightPx = header.getBoundingClientRect().height;
		}
		if ($uploadedImgBase64 === null || $uploadedImgFileName === '') {
			goto(base == '' ? '/' : base);
		}
		uploadedImage = $uploadedImgBase64;
		window.addEventListener('resize', () => {
			if (imageCanvas) {
				const canvasElementSize = imageCanvas.getBoundingClientRect();
				ImgResToCanvasSizeRatio = imageCanvas.width / canvasElementSize.width;
			}
		});
		if (!uploadedImage || !uploadedImgFileName) {
			goto(base == '' ? '/' : base);
			return;
		}
		await setupEditor(uploadedImage, $uploadedImgFileName);

		if (!$mainWorker) {
			goto(base == '' ? '/' : base);
		} else {
			/*mainworker has set modelsLoaded function callback (setups editor etc.)
			it is called either based on response to this message or after models are loaded 
			(then, the message is sent automatically after loading)*/
			$mainWorker.postMessage({ type: MESSAGE_TYPES.CHECK_MODELS_LOADING_STATE });
		}
		panzoom = Panzoom(canvasesContainer, {
			disablePan: !enablePan,
			minScale: 1,
			maxScale: 10,
			disableZoom: anythingEssentialLoading
		}) as PanzoomObject;
		canvasesContainer.parentElement!.addEventListener('wheel', panzoom.zoomWithWheel);
		canvasesContainer.addEventListener('panzoomchange', (event: any) => {
			currentZoom = event.detail.scale;
		});
		setTimeout(() => panzoom.zoom(0, 100));
		panzoom.zoom(5, { animate: true });
	});
	const drawerStore = getDrawerStore();

	let panzoom: any;
	let currentZoom = 1;

	function handlePanzoomSettingsChange(enablePan: boolean, anythingEssentialLoading: boolean) {
		if (panzoom) {
			panzoom.setOptions({
				disablePan: !enablePan,
				disableZoom: anythingEssentialLoading
			});
		}
	}

	$: handlePanzoomSettingsChange(enablePan, anythingEssentialLoading);
</script>

<Drawer>
	<h2 class="p-4 font-semibold">Smart object remover</h2>
	<hr />
	<TabGroup>
		<Tab
			class="px-8 py-4"
			bind:group={selectedTool}
			name="segment_anything"
			value="segment_anything"
		>
			<span class="flex gap-x-2 items-center"> <WandSparkles size={18} /> Smart selector</span>
		</Tab>
		<Tab class="px-8 py-4" bind:group={selectedTool} name="brush" value="brush">
			<span class="flex gap-x-2 items-center"> <Brush size={18} /> Brush</span>
		</Tab>

		<div slot="panel" class="p-4">
			{#if selectedTool === 'brush'}
				<div>
					<label for="brushtoolselect" class="font-semibold">Select tool:</label>
					<div class="font-thin">
						Select brush or eraser tool to mark the area you want to remove
					</div>
					<RadioGroup class="text-token mb-4" id="brushtoolselect">
						<RadioItem
							bind:group={selectedBrushMode}
							on:change={(e) => handleBrushModeChange(e, maskCanvas)}
							name="brushtool"
							value="brush"
						>
							<Brush size={18} />
						</RadioItem>
						<RadioItem
							bind:group={selectedBrushMode}
							on:change={(e) => handleBrushModeChange(e, maskCanvas)}
							name="brushtool"
							value="eraser"
						>
							<Eraser size={18} />
						</RadioItem>
					</RadioGroup>

					<label for="brushSize">Brush size: {brushSize}</label>
					<input type="range" min="1" max="500" bind:value={brushSize} />
				</div>
			{:else if selectedTool === 'segment_anything'}
				<label for="sammodeselect" class="font-semibold">Select smart selector mode:</label>
				<div class="font-thin">
					Add positive (adds area to selection) or negative (removes area from selection) points for
					selection
				</div>
				<RadioGroup id="sammodeselect" class="text-token mb-4">
					<RadioItem bind:group={selectedSAMMode} name="sammode" value="positive">
						<PlusCircleIcon size={18} />
					</RadioItem>
					<RadioItem bind:group={selectedSAMMode} name="sammode" value="negative">
						<MinusCircle size={18} />
					</RadioItem>
				</RadioGroup>
				<label for="pixelsDilatation">Mask dilatation: {pixelsDilatation}</label>
				<div class="font-thin">For best results, all edges of object should be inside the mask</div>
				<input type="range" min="0" max="25" bind:value={pixelsDilatation} />
			{/if}
		</div>
	</TabGroup>
</Drawer>

<AppShell slotSidebarLeft="overflow-visible lg:w-80 w-0 h-screen shadow-md z-50">
	<svelte:fragment slot="header">
		<AppBar id="appbar">
			<svelte:fragment slot="lead">
				<button class="lg:hidden btn btn-sm mr-4" on:click={() => drawerStore.open({})}>
					<span>
						<svg viewBox="0 0 100 80" class="fill-token w-4 h-4">
							<rect width="100" height="20" />
							<rect y="30" width="100" height="20" />
							<rect y="60" width="100" height="20" />
						</svg>
					</span>
				</button>
				<a href={base == '' ? '/' : base} class="font-bold lg:inline hidden">Smart Object remover</a
				>
			</svelte:fragment>
			<svelte:fragment slot="trail">
				<a href={base == '' ? '/' : base} class="font-semibold">Home</a>
				<a href="{base}/about" class="font-semibold">About</a>
				<LightSwitch />
			</svelte:fragment>
		</AppBar>
	</svelte:fragment>
	<svelte:fragment slot="sidebarLeft">
		<TabGroup>
			<Tab
				class="px-8 py-4"
				bind:group={selectedTool}
				name="segment_anything"
				value="segment_anything"
			>
				<span class="flex gap-x-2 items-center"> <WandSparkles size={18} /> Smart selector</span>
			</Tab>
			<Tab class="px-8 py-4" bind:group={selectedTool} name="brush" value="brush">
				<span class="flex gap-x-2 items-center"> <Brush size={18} /> Brush</span>
			</Tab>

			<div slot="panel" class="p-4">
				{#if selectedTool === 'brush'}
					<div>
						<label for="brushtoolselect" class="font-semibold">Select tool:</label>
						<div class="font-thin">
							Select brush or eraser tool to mark the area you want to remove
						</div>
						<RadioGroup class="text-token mb-4" id="brushtoolselect">
							<RadioItem
								bind:group={selectedBrushMode}
								on:change={(e) => handleBrushModeChange(e, maskCanvas)}
								name="brushtool"
								value="brush"
							>
								<Brush size={18} />
							</RadioItem>
							<RadioItem
								bind:group={selectedBrushMode}
								on:change={(e) => handleBrushModeChange(e, maskCanvas)}
								name="brushtool"
								value="eraser"
							>
								<Eraser size={18} />
							</RadioItem>
						</RadioGroup>

						<label for="brushSize">Brush size: {brushSize}</label>
						<input type="range" min="1" max="500" bind:value={brushSize} />
					</div>
				{:else if selectedTool === 'segment_anything'}
					<label for="sammodeselect" class="font-semibold">Select smart selector mode:</label>
					<div class="font-thin">
						Add positive (adds area to selection) or negative (removes area from selection) points
						for selection
					</div>
					<RadioGroup id="sammodeselect" class="text-token mb-4">
						<RadioItem bind:group={selectedSAMMode} name="sammode" value="positive">
							<PlusCircleIcon size={18} />
						</RadioItem>
						<RadioItem bind:group={selectedSAMMode} name="sammode" value="negative">
							<MinusCircle size={18} />
						</RadioItem>
					</RadioGroup>
					<label for="pixelsDilatation">Mask dilatation: {pixelsDilatation}</label>
					<div class="font-thin">
						For best results, all edges of object should be inside the mask
					</div>
					<input type="range" min="0" max="25" bind:value={pixelsDilatation} />
				{/if}
			</div>
		</TabGroup>
	</svelte:fragment>
	<div
		class="
	flex flex-col gap-y-4
	2xl:px-64 xl:px-16 md:px-8 px-2 py-4
	"
		style="max-height: calc(100vh - {headerHeightPx}px)"
	>
		<!-- top buttons panel -->
		<div class="flex flex-none justify-between">
			<!-- left buttons -->
			<div class="flex lg:gap-x-2 gap-x-1">
				<button
					class="btn btn-sm lg:btn-md variant-filled"
					on:click={undoLastAction}
					disabled={editorStatesHistory.length === 0 || anythingEssentialLoading}
					><Undo class="lg:w-6 lg:h-6 w-4 h-4" /></button
				>
				<button
					class="btn btn-sm lg:btn-md variant-filled"
					on:click={redoLastAction}
					disabled={editorStatesUndoed.length === 0 || anythingEssentialLoading}
					><Redo class="lg:w-6 lg:h-6 w-4 h-4" /></button
				>
				<button
					disabled={anythingEssentialLoading}
					class="btn btn-sm lg:btn-md variant-filled"
					on:click={reset}><RotateCw class="lg:w-6 lg:h-6 w-4 h-4" /></button
				>
			</div>

			<!-- right buttons -->
			<div class="flex lg:gap-x-2 gap-x-1">
				
				<button
					disabled={anythingEssentialLoading}
					class="btn btn-sm lg:btn-md variant-filled select-none"
					on:mousedown={() => {
						maskCanvas.style.display = 'none';
						imageCanvas.style.display = 'none';
						originalImgElement.style.display = 'block';
					}}
					on:touchstart={() => {
						maskCanvas.style.display = 'none';
						imageCanvas.style.display = 'none';
						originalImgElement.style.display = 'block';
					}}
					on:mouseup={() => {
						maskCanvas.style.display = 'block';
						imageCanvas.style.display = 'block';
						originalImgElement.style.display = 'none';
					}}
					on:touchend={() => {
						maskCanvas.style.display = 'block';
						imageCanvas.style.display = 'block';
						originalImgElement.style.display = 'none';
					}}
				>
					<span class="hidden sm:inline no-margin">Hold to compare</span>
					<span class="no-margin sm:hidden">Compare</span>
				</button>

				<button
					disabled={anythingEssentialLoading}
					class="btn btn-sm lg:btn-md variant-filled"
					on:click={() => downloadImage(currentEditorState.imgData)}
				>
					<span class="flex gap-x-2 items-center"
						><HardDriveDownload class="lg:w-6 lg:h-6 w-4 h-4" /> Download
					</span>
				</button>
			</div>
		</div>
		<!-- editor canvases-->
		<div id="mainEditorContainer" class="grow overflow-hidden"
			style="cursor: {anythingEssentialLoading ? 'not-allowed' : 'default'}"
		>
			<div
				class="absolute w-full h-full flex items-center flex-col gap-y-2 justify-center z-10 {!anythingEssentialLoading
					? 'hidden'
					: ''}"
			>
				<ProgressRadial meter="stroke-primary-500" track="stroke-primary-500/30" />
				{#if encoderLoading || decoderLoading}
					<div class="text-primary-500 font-bold text-2xl mt-2">Models loading...</div>
				{:else if inpaintingRunning}
					<div class="text-primary-500 font-bold text-2xl mt-2">Removing area...</div>
				{:else if isEmbedderRunning}
					<div class="text-primary-500 font-bold text-2xl mt-2">Processing new image...</div>
				{:else if decoderRunning}
					<div class="text-primary-500 font-bold text-2xl mt-2">Computing mask...</div>
				{/if}
			</div>
			<div
				class="canvases w-full"
				bind:this={canvasesContainer}
				style="cursor: {enablePan
					? 'move'
					: selectedTool === 'segment_anything'
					? 'default'
					: currentCursor === 'default'
					? 'auto'
					: 'none'}
				"
			>
				<div
					class="relative !h-full !w-full"
					on:mouseenter={!enablePan ? showBrushCursor : undefined}
					on:mouseleave={!enablePan ? hideBrushCursor : undefined}
					role="group"
				>
					<div class="absolute w-full h-full overflow-hidden">
						<div
							id="brushToolCursor"
							style="
		display: {selectedTool === 'segment_anything' || anythingEssentialLoading
								? 'none'
								: currentCursor === 'default'
								? 'none'
								: 'block'};
		width: {brushSize - 2}px;
		height: {brushSize - 2}px;
		left: {currentCanvasRelativeX}px;
		top: {currentCanvasRelativeY}px;
		background-color: {selectedBrushMode === 'brush' ? '#408dff' : '#f5f5f5'};
		border: 1px solid #0261ed;
		opacity: {isPainting ? 0.6 : 0.5};

	"
						/>
					</div>
					<!-- default width so the page isnt empty till load -->
					<canvas
						class="shadow-lg
						{anythingEssentialLoading ? 'opacity-30 cursor-not-allowed' : ''} "
						id="imageCanvas"
						bind:this={imageCanvas}
					/>

					<canvas
						id="maskCanvas"
						class="
						{enablePan ? '' : 'panzoom-exclude'} 
						{anythingEssentialLoading ? 'opacity-30 cursor-not-allowed' : ''}"
						bind:this={maskCanvas}
						on:mousedown={selectedTool === 'brush' && !enablePan
							? (e) => startPaintingMouse(e, maskCanvas)
							: undefined}
						on:mouseup={selectedTool === 'brush' && !enablePan ? stopPainting : undefined}
						on:mousemove={(event) =>
							selectedTool === 'brush' && !enablePan
								? handleEditorMouseMove(event, maskCanvas)
								: undefined}
						on:touchstart={selectedTool === 'brush'
							? // && !enablePan
							  (e) => {
									console.log('start');
									console.log(e);
									if (e.touches.length === 1) {
										startPaintingTouch(e, maskCanvas);
									} else {
										stopPainting();
									}
							  }
							: undefined}
						on:touchend={selectedTool === 'brush'
							? // &&  !enablePan
							  (e) => {
									console.log('end');
									console.log(e);
									stopPainting();
							  }
							: undefined}
						on:touchmove={selectedTool === 'brush'
							? // &&  !enablePan
							  (e) => {
									console.log('move');
									console.log(e);
									if (e.touches.length === 1) {
										e.preventDefault();
										handleEditorTouchMove(e, maskCanvas);
									}
							  }
							: undefined}
						on:click={selectedTool === 'segment_anything' && !enablePan
							? async (e) => {
									console.log(e);
									handleCanvasClick(e);
							  }
							: undefined}
					/>

					<img
						class="shadow-lg
						{anythingEssentialLoading ? 'opacity-50 cursor-not-allowed' : ''}
						"
						src={$uploadedImgBase64}
						alt="originalImage"
						bind:this={originalImgElement}
					/>
				</div>
			</div>
		</div>
		<!-- bottom buttons -->
		<div class="flex flex-none flex-wrap lg:gap-x-2 gap-x-1">
			<div class="sm:flex-1 flex-0" />
			<div
				class="btn-group variant-filled {anythingEssentialLoading
					? 'opacity-50 cursor-not-allowed'
					: ''}"
			>
				<div
					class="flex items-center justify-center {anythingEssentialLoading
						? 'cursor-not-allowed'
						: ''}"
				>
					<input
						type="checkbox"
						id="choose-me"
						class="peer hidden"
						disabled={anythingEssentialLoading}
						bind:checked={enablePan}
					/>

					<label
						for="choose-me"
						class="select-none {anythingEssentialLoading
							? 'cursor-not-allowed opacity-50'
							: 'cursor-pointer'} rounded-lg
					  peer-checked:bg-surface-700 dark:peer-checked:bg-surface-200 peer-checked:border-gray-200 !px-1 !mx-1 !ml-2 lg:!p-2 lg:!mr-2 lg:!ml-3"
					>
						<MoveIcon />
					</label>
				</div>
				<button
					on:click={(e) => e.preventDefault()}
					disabled={anythingEssentialLoading}
					class="cursor-default !px-1 lg:!px-3"
				>
					<div class="flex items-center justify-center gap-x-2">
						<button
							on:click={(e) => panzoom.zoomOut()}
							class="{anythingEssentialLoading
								? 'cursor-not-allowed'
								: 'cursor-pointer'} btn btn-sm !p-0 lg:!px-4 lg:!py-2"
						>
							<MinusIcon />
						</button>

						<span
							class="border-2 lg:text-md text-sm font-medium rounded-md p-2 border-surface-600 dark:border-surface-700"
							>{Math.round(currentZoom * 100)} %</span
						>
						<button
							on:click={(e) => panzoom.zoomIn()}
							class="{anythingEssentialLoading
								? 'cursor-not-allowed'
								: 'cursor-pointer'} btn btn-sm variant-filled !p-0 lg:!px-4 lg:!py-2"
						>
							<PlusIcon />
						</button>
					</div>
				</button>
				<button
					disabled={anythingEssentialLoading}
					on:click={() => panzoom.reset()}
					class="!px-2 !pr-3 lg:!px-4 lg:!pr-5"
				>
					<ScanEyeIcon />
				</button>
			</div>
			<div class="flex-1 flex justify-end">
				<button
					class="btn lg:btn-xl md:btn-md btn-sm variant-filled-primary text-white dark:text-white font-semibold"
					disabled={anythingEssentialLoading || inpainterLoading}
					on:click={async () => handleInpainting()}
				>
					<span class="flex gap-x-2 items-center">
						{#if inpainterLoading && !anythingEssentialLoading}
							<ProgressRadial
								width="w-6"
								meter="stroke-white"
								track="stroke-white/30"
								value={undefined}
							/>
							<span class="hidden lg:inline"> Loading model</span>
						{:else}
							<WandSparkles size={18} />
							Remove
						{/if}
					</span>
				</button>
				<!-- <div class="w-20 h-20 ml-auto">


			  </div> -->
			</div>
		</div>
	</div>
</AppShell>

<style>
	.canvases canvas,
	.canvases img {
		inset: 0;
		display: block;
		width: 100%;
		height: 100%;
		object-fit: contain;
	}

	.canvases #maskCanvas {
		position: absolute;
	}
	.canvases #maskCanvas {
		opacity: 0.5;
	}
	#mainEditorContainer {
		position: relative;
		display: flex;
		justify-content: center;
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
	.no-margin {
		margin: 0 !important;
	}
</style>
