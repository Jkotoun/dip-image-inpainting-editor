<script lang="ts">
	import { onMount } from 'svelte';
	import { base } from '$app/paths';
	import {
		renderEditorState,
		type editorState,
		type SAMmarker,
		canvasBrushDraw,
		runInpainting,
		createEmptyMaskArray,
		createDecoderInputDict,
		runModelEncoder
	} from './editorModule';
	import { getResizedImgData } from '$lib/onnxHelpers';
	import { RGB_CHW_array_to_imageData } from './editorModule';
	import {
		dilateMaskByPixels,
		maskArrayFromImgData,
		downloadImage,
		clearCanvas
	} from '$lib/editorHelpers';
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
	import Navbar from '../../components/navbar.svelte';

	let uploadedImage: string | null = null;
	let headerHeightPx = 0;
	let encoderLoading = true;
	let decoderLoading = true;
	let inpainterLoading = true;
	let isEncoderRunning = false;
	let decoderRunning = false;
	let inpaintingRunning = false;
	$: anythingEssentialLoading =
		inpaintingRunning || encoderLoading || decoderLoading || isEncoderRunning || decoderRunning;

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

	//editor state management
	let imgDataOriginal: ImageData;
	let imgName: string = 'default';
	let currentEditorState: editorState;
	let editorStatesHistory: editorState[] = [];
	//saved for potential redos after undo actions, emptied on new action after series of undos
	let editorStatesUndoed: editorState[] = [];
	//brush tool
	let brushSize = 10;
	let prevMouseX = 0;
	let prevMouseY = 0;
	let currentCanvasRelativeX = 0;
	let currentCanvasRelativeY = 0;
	let selectedBrushMode: 'brush' | 'eraser' = 'brush'; // Initial selected option
	let selectedTool: tool = 'segment_anything';
	let displayBrushCursor: boolean = false;
	let selectedSAMMode: 'positive' | 'negative' = 'positive';
	const drawerStore = getDrawerStore();
	type tool = 'brush' | 'segment_anything';

	const currentCanvasCursor = (enablePan: boolean, selectedTool: tool) => {
		if (enablePan === true) {
			return 'move';
		} else {
			if (selectedTool === 'segment_anything') {
				return 'default';
			} else {
				return 'none';
			}
		}
	};
	let panzoom: any;
	let currentZoom = 1;

	// change dilatation when pixelsDilatation value changes
	$: changeDilatation(pixelsDilatation, currentEditorState);

	if ($mainWorker) {
		$mainWorker.onmessage = handleWorkerModelsMessages;
	}

	const runEncoderCurrentState = async () => {
		getResizedImgData(currentEditorState.imgData, longSideLength).then((resizedImgRGBData) => {
			//set globals for decoder (needs to map input points to correct coordinates in resized image)
			resizedImgWidth = resizedImgRGBData.width;
			resizedImgHeight = resizedImgRGBData.height;
			runModelEncoder(resizedImgRGBData, $mainWorker!);
		});
	};

	function handleWorkerModelsMessages(event: MessageEvent<any>) {
		const { data, type } = event.data;
		if (type === MESSAGE_TYPES.INPAINTER_LOADED) {
			inpainterLoading = false;
		}
		if (type === MESSAGE_TYPES.DECODER_RUN_RESULT_SUCCESS && !decoderLoading) {
			const SAMMaskArray = data.map((val: number[]) => val.map((v) => v > 0.0));
			if (SAMMaskArray) {
				dilateMaskByPixels(pixelsDilatation, SAMMaskArray).then((dilatedMask) => {
					currentEditorState.maskSAMDilated = dilatedMask;
					currentEditorState.maskSAM = SAMMaskArray;
					renderEditorState(currentEditorState, imageCanvas, maskCanvas, ImgResToCanvasSizeRatio);
				});
			}
			decoderRunning = false;
		} else if (type === MESSAGE_TYPES.ENCODER_RUN_RESULT_SUCCESS && !encoderLoading) {
			currentEditorState.currentImgEmbedding = {
				data: data.embeddings,
				dims: data.dims
			};
			isEncoderRunning = false;
		} else if (type === MESSAGE_TYPES.INPAINTING_RUN_RESULT_SUCCESS && !inpainterLoading) {
			RGB_CHW_array_to_imageData(data, imageCanvas.height, imageCanvas.width).then(
				(resultImgData) => {
					handleInpaintedImgData(resultImgData);
					inpaintingRunning = false;
				}
			);
		} else if (type === MESSAGE_TYPES.ALL_MODELS_LOADED) {
			decoderLoading = encoderLoading = inpainterLoading = false;
			if (
				currentEditorState &&
				currentEditorState.imgData &&
				!isEncoderRunning &&
				!currentEditorState.currentImgEmbedding
			) {
				isEncoderRunning = true;
				runEncoderCurrentState();
			}
		} else if (type === MESSAGE_TYPES.ENCODER_DECODER_LOADED) {
			encoderLoading = decoderLoading = false;
			if (
				currentEditorState &&
				currentEditorState.imgData &&
				!isEncoderRunning &&
				!currentEditorState.currentImgEmbedding
			) {
				isEncoderRunning = true;
				runEncoderCurrentState().then((_) => {
					$mainWorker?.postMessage({ type: MESSAGE_TYPES.LOAD_INPAINTER });
				});
			}
		}
	}

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
		currentEditorState = await initEditorState(uploadedImage, $uploadedImgFileName);

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
			disableZoom: anythingEssentialLoading,
			cursor: 'default'
		}) as PanzoomObject;
		canvasesContainer.parentElement!.addEventListener('wheel', panzoom.zoomWithWheel);
		canvasesContainer.addEventListener('panzoomchange', (event: any) => {
			currentZoom = event.detail.scale;
		});
		setTimeout(() => panzoom.zoom(0, 100));
		panzoom.zoom(5, { animate: true });
	});

	//change dilatation when pixelsDilatation value changes
	async function changeDilatation(pixelsDilatation: number, currentEditorState: editorState) {
		if (currentEditorState) {
			currentEditorState.maskSAMDilated = await dilateMaskByPixels(
				pixelsDilatation,
				currentEditorState.maskSAM
			);
			renderEditorState(currentEditorState, imageCanvas, maskCanvas, ImgResToCanvasSizeRatio);
		}
	}

	//initializes editor state on new image
	const initEditorState = async (sourceImgBase64Data: string, sourceImgName: string) => {
		return new Promise<editorState>((resolve, reject) => {
			const img = new Image();
			img.src = sourceImgBase64Data;
			imgName = sourceImgName;
			img.onload = async () => {
				// Calculate aspect ratio
				imageCanvas.width = img.width;
				maskCanvas.width = img.width;
				imageCanvas.height = img.height;
				maskCanvas.height = img.height;
				const canvasElementSize = imageCanvas.getBoundingClientRect();
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
				let newEditorState = {
					maskBrush: createEmptyMaskArray(imgDataOriginal.width, imgDataOriginal.height),
					maskSAM: createEmptyMaskArray(imgDataOriginal.width, imgDataOriginal.height),
					maskSAMDilated: createEmptyMaskArray(imgDataOriginal.width, imgDataOriginal.height),
					clickedPositions: new Array<SAMmarker>(),
					imgData: imgDataOriginal,
					currentImgEmbedding: undefined
				} as editorState;
				resolve(newEditorState);
			};

			img.onerror = (error) => {
				reject(error);
			};
		});
	};

	//post message with decoder input dict to webworker
	const runDecoderCurrentState = async () => {
		if (!currentEditorState.currentImgEmbedding) {
			return;
		}
		decoderRunning = true;
		const modelInput = await createDecoderInputDict(
			currentEditorState,
			resizedImgWidth,
			resizedImgHeight
		);
		$mainWorker?.postMessage({
			type: MESSAGE_TYPES.DECODER_RUN,
			data: modelInput
		});
	};

	async function handleCanvasClick(event: MouseEvent) {
		console.log('canvas log');
		console.log(imageCanvas.getBoundingClientRect().width);
		if (anythingEssentialLoading) {
			return;
		}
		//it logs -0 at 0,0 for some reason
		event.preventDefault();
		const xScaled = Math.abs(event.offsetX) * ImgResToCanvasSizeRatio;
		const yScaled = Math.abs(event.offsetY) * ImgResToCanvasSizeRatio;

		console.log(event.offsetX, event.offsetY);
		console.log('scaled');
		console.log(xScaled, yScaled);

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
		renderEditorState(currentEditorState, imageCanvas, maskCanvas, ImgResToCanvasSizeRatio);
		runDecoderCurrentState();
	}

	function undoLastEditorAction() {
		//new state management
		if (editorStatesHistory.length > 0 && currentEditorState) {
			editorStatesUndoed = [...editorStatesUndoed, currentEditorState];
			currentEditorState = editorStatesHistory[editorStatesHistory.length - 1];
			editorStatesHistory = editorStatesHistory.slice(0, -1);
			renderEditorState(currentEditorState, imageCanvas, maskCanvas, ImgResToCanvasSizeRatio);
		}
	}

	function redoLastEditorAction() {
		//new state management
		if (editorStatesUndoed.length > 0 && currentEditorState) {
			editorStatesHistory = [...editorStatesHistory, currentEditorState];
			currentEditorState = editorStatesUndoed[editorStatesUndoed.length - 1];
			editorStatesUndoed = editorStatesUndoed.slice(0, -1);
			renderEditorState(currentEditorState, imageCanvas, maskCanvas, ImgResToCanvasSizeRatio);
		}
	}

	function resetEditorState() {
		//clear all states
		panzoom.reset();
		currentEditorState = {
			maskBrush: createEmptyMaskArray(imgDataOriginal.width, imgDataOriginal.height),
			maskSAM: createEmptyMaskArray(imgDataOriginal.width, imgDataOriginal.height),
			maskSAMDilated: createEmptyMaskArray(imgDataOriginal.width, imgDataOriginal.height),
			clickedPositions: [],
			imgData: imgDataOriginal,
			currentImgEmbedding: editorStatesHistory[0]
				? editorStatesHistory[0].currentImgEmbedding
				: currentEditorState.currentImgEmbedding
		};
		editorStatesHistory = [];
		editorStatesUndoed = [];
		renderEditorState(currentEditorState, imageCanvas, maskCanvas, ImgResToCanvasSizeRatio);
	}

	// brush tool
	function startPaintingMouse(event: MouseEvent, canvas: HTMLCanvasElement) {
		isPainting = true;
		prevMouseX = event.offsetX * ImgResToCanvasSizeRatio;
		prevMouseY = event.offsetY * ImgResToCanvasSizeRatio;
		handleEditorCursorMove(event, canvas);
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
	}

	function stopPainting() {
		isPainting = false;
		const ctx = maskCanvas.getContext('2d', { willReadFrequently: true });
		if (!ctx) {
			console.error('canvas context error');
			return;
		}
		const imageData = ctx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
		let maskArray: boolean[][] = maskArrayFromImgData(
			imageData,
			maskCanvas.width,
			maskCanvas.height
		);
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
	function handleEditorCursorMove(event: MouseEvent | TouchEvent, canvas: HTMLCanvasElement) {
		let x: number, y: number;
		if (event instanceof MouseEvent) {
			x = event.offsetX * ImgResToCanvasSizeRatio;
			y = event.offsetY * ImgResToCanvasSizeRatio;
			currentCanvasRelativeX = event.offsetX;
			currentCanvasRelativeY = event.offsetY;
		} else {
			let touch = event.touches[0]; // Get the first touch, you might handle multi-touch differently
			let targetRect = canvas.getBoundingClientRect(); // Get the target element's position

			// Calculate offsetX and offsetY
			let offsetX = touch.clientX - targetRect.left;
			let offsetY = touch.clientY - targetRect.top;
			x = offsetX * ImgResToCanvasSizeRatio;
			y = offsetY * ImgResToCanvasSizeRatio;
			currentCanvasRelativeX = offsetX;
			currentCanvasRelativeY = offsetY;
		}
		if (isPainting) {
			let canvasCtx = canvas.getContext('2d', { willReadFrequently: true });
			if (!canvasCtx) {
				console.error('canvas context error');
				return;
			}
			const { currentX, currentY } = canvasBrushDraw(
				x,
				y,
				prevMouseX,
				prevMouseY,
				brushSize,
				canvasCtx,
				ImgResToCanvasSizeRatio
			);
			prevMouseX = currentX;
			prevMouseY = currentY;
		}
	}

	function getImageData(canvas: HTMLCanvasElement) {
		return canvas.getContext('2d')!.getImageData(0, 0, canvas.width, canvas.height);
	}

	function handleBrushModeChange(event: any, maskCanvas: HTMLCanvasElement) {
		selectedBrushMode = event.target.value;
		// Get the canvas element
		const context = maskCanvas.getContext('2d');
		if (!context) return;

		context.globalCompositeOperation =
			selectedBrushMode === 'brush' ? 'source-over' : 'destination-out';
	}
	async function handleInpainting() {
		if (!anythingEssentialLoading) {
			inpaintingRunning = true;
			runInpainting(currentEditorState, $mainWorker!);
		}
	}

	//handle inpainted image data - set and render new editor state and run encoder on new image
	async function handleInpaintedImgData(inpaintedImgData: ImageData) {
		editorStatesHistory = [...editorStatesHistory, currentEditorState];
		let newEditorState: editorState = {
			maskBrush: createEmptyMaskArray(imgDataOriginal.width, imgDataOriginal.height),
			maskSAM: createEmptyMaskArray(imgDataOriginal.width, imgDataOriginal.height),
			maskSAMDilated: createEmptyMaskArray(imgDataOriginal.width, imgDataOriginal.height),
			clickedPositions: [],
			imgData: inpaintedImgData,
			currentImgEmbedding: undefined
		};
		currentEditorState = newEditorState;
		renderEditorState(currentEditorState, imageCanvas, maskCanvas, ImgResToCanvasSizeRatio);
		isEncoderRunning = true;
		runEncoderCurrentState();
	}

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
				<input
					type="range"
					min="0"
					max="25"
					bind:value={pixelsDilatation}
					on:change={(e) => console.log(e)}
				/>
			{/if}
		</div>
	</TabGroup>
</Drawer>

<AppShell slotSidebarLeft="overflow-visible lg:w-80 w-0 h-screen shadow-md z-50">
	<svelte:fragment slot="header">
		<Navbar
			basepath={base}
			navTitle={{ name: 'Smart Object Remover', href: base == '' ? '/' : base }}
			links={[
				{ name: 'Home', href: base == '' ? '/' : base },
				{ name: 'About', href: `${base}/about` }
			]}
			drawerMenu
			on:openDrawerMenu={() => drawerStore.open({})}
		/>
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
					on:click={undoLastEditorAction}
					disabled={editorStatesHistory.length === 0 || anythingEssentialLoading}
					><Undo class="lg:w-6 lg:h-6 w-4 h-4" /></button
				>
				<button
					class="btn btn-sm lg:btn-md variant-filled"
					on:click={redoLastEditorAction}
					disabled={editorStatesUndoed.length === 0 || anythingEssentialLoading}
					><Redo class="lg:w-6 lg:h-6 w-4 h-4" /></button
				>
				<button
					disabled={anythingEssentialLoading}
					class="btn btn-sm lg:btn-md variant-filled"
					on:click={resetEditorState}><RotateCw class="lg:w-6 lg:h-6 w-4 h-4" /></button
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
					on:click={() => downloadImage(currentEditorState.imgData, imgName)}
				>
					<span class="flex gap-x-2 items-center"
						><HardDriveDownload class="lg:w-6 lg:h-6 w-4 h-4" /> Download
					</span>
				</button>
			</div>
		</div>
		<!-- editor canvases-->
		<div
			id="mainEditorContainer"
			class="grow overflow-hidden"
			style="cursor: {anythingEssentialLoading ? 'not-allowed' : 'default'}"
		>
			<div
				class="absolute !w-full !h-full flex items-center flex-col gap-y-2 justify-center z-10 {!anythingEssentialLoading
					? 'hidden'
					: ''}"
			>
				<ProgressRadial meter="stroke-primary-500" track="stroke-primary-500/30" />
				{#if encoderLoading || decoderLoading}
					<div class="text-primary-500 font-bold text-2xl mt-2">Models loading...</div>
				{:else if inpaintingRunning}
					<div class="text-primary-500 font-bold text-2xl mt-2">Removing area...</div>
				{:else if isEncoderRunning}
					<div class="text-primary-500 font-bold text-2xl mt-2">Processing new image...</div>
				{:else if decoderRunning}
					<div class="text-primary-500 font-bold text-2xl mt-2">Computing mask...</div>
				{/if}
			</div>
			<div class="canvases w-full" bind:this={canvasesContainer}>
				<div class="relative !h-full !w-full flex justify-center" role="group">
					<div class="flex-none" />
					<!-- default width so the page isnt empty till load -->
					<div
						class="relative flex-none shrink"
						style="cursor: {currentCanvasCursor(enablePan, selectedTool)}"
						on:mouseenter={() => {
							displayBrushCursor =
								selectedTool === 'brush' && !anythingEssentialLoading && !enablePan;
						}}
						on:mouseleave={() => {
							displayBrushCursor = false;
						}}
						role="group"
					>
						<div class="absolute w-full h-full overflow-hidden">
							<div
								id="brushToolCursor"
								role="group"
								style="
			display: {displayBrushCursor ? 'block' : 'none'};
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
									? handleEditorCursorMove(event, maskCanvas)
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
										if (e.touches.length === 1) {
											e.preventDefault();
											handleEditorCursorMove(e, maskCanvas);
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
					<div class="flex-none" />
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
