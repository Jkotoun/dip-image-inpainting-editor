<script lang="ts">
	import { onMount } from 'svelte';
	import { base } from '$app/paths';
	import type { tool, brushMode, SAMMode } from '../../types/editorTypes';
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
		WandSparkles,
		Undo,
		Redo,
		RotateCw,
		HardDriveDownload,
	} from 'lucide-svelte';
	import {
		AppShell,
		Drawer,
		getDrawerStore,
		ProgressRadial,
	} from '@skeletonlabs/skeleton';
	import { goto } from '$app/navigation';
	import Navbar from './../../components/Navbar.svelte';
	import EditorToolSelection from './../../components/EditorToolSelection.svelte';
	import PanzoomCanvasControls from '../../components/PanzoomCanvasControls.svelte';

	let gUploadedImage: string | null = null;
	let gHeaderHeightPx = 0;
	let gEncoderLoading = true;
	let gDecoderLoading = true;
	let gInpainterLoading = true;
	let gIsEncoderRunning = false;
	let gDecoderRunning = false;
	let gInpaintingRunning = false;
	let gIsPainting = false;

	//segmentation model constants
	const gLongSideLength = 1024;
	let gResizedImgWidth: number;
	let gResizedImgHeight: number;

	let gImgResToCanvasSizeRatio: number = 1;
	let gImageCanvas: any;
	let gCanvasesContainer: any;
	let gMaskCanvas: any;
	let gOriginalImgElement: HTMLImageElement;

	//editor state management
	let gImgDataOriginal: ImageData;
	let gImgName: string = 'default';
	let gCurrentEditorState: editorState;
	let gEditorStatesHistory: editorState[] = [];
	//saved for potential redos after undo actions, emptied on new action after series of undos
	let gEditorStatesUndoed: editorState[] = [];
	//brush tool
	let gPrevMouseX = 0;
	let gPrevMouseY = 0;
	let gCurrentCanvasRelativeX = 0;
	let gCurrentCanvasRelativeY = 0;
	let gDisplayBrushCursor: boolean = false;
	const drawerStore = getDrawerStore();

	let gSAMMaskDilatation: number;
	let gSelectedBrushMode: brushMode; // Initial selected option
	let gSelectedTool: tool;
	let gSelectedSAMMode: SAMMode;
	let gBrushToolSize: number;
	
	let gPanEnabled: boolean = false;
	let gCurrentZoom: number = 1;
	let gPanzoomObj: any;
	let gShowOriginalImage: boolean = false;
	
	$: handleBrushModeChange(gSelectedBrushMode, gMaskCanvas);
	$: changeDilatation(gSAMMaskDilatation, gCurrentEditorState);
	$: handlePanzoomSettingsChange(gPanEnabled, gAnythingEssentialLoading);
	$: gAnythingEssentialLoading =
		gInpaintingRunning || gEncoderLoading || gDecoderLoading || gIsEncoderRunning || gDecoderRunning;

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

	// change dilatation when pixelsDilatation value changes

	if ($mainWorker) {
		$mainWorker.onmessage = handleWorkerModelsMessages;
	}

	const runEncoderCurrentState = async () => {
		getResizedImgData(gCurrentEditorState.imgData, gLongSideLength).then((resizedImgRGBData) => {
			//set globals for decoder (needs to map input points to correct coordinates in resized image)
			gResizedImgWidth = resizedImgRGBData.width;
			gResizedImgHeight = resizedImgRGBData.height;
			runModelEncoder(resizedImgRGBData, $mainWorker!);
		});
	};

	function handleWorkerModelsMessages(event: MessageEvent<any>) {
		const { data, type } = event.data;
		if (type === MESSAGE_TYPES.INPAINTER_LOADED) {
			gInpainterLoading = false;
		}
		if (type === MESSAGE_TYPES.DECODER_RUN_RESULT_SUCCESS && !gDecoderLoading) {
			const SAMMaskArray = data.map((val: number[]) => val.map((v) => v > 0.0));
			if (SAMMaskArray) {
				dilateMaskByPixels(gSAMMaskDilatation, SAMMaskArray).then((dilatedMask) => {
					gCurrentEditorState.maskSAMDilated = dilatedMask;
					gCurrentEditorState.maskSAM = SAMMaskArray;
					renderEditorState(gCurrentEditorState, gImageCanvas, gMaskCanvas, gImgResToCanvasSizeRatio);
				});
			}
			gDecoderRunning = false;
		} else if (type === MESSAGE_TYPES.ENCODER_RUN_RESULT_SUCCESS && !gEncoderLoading) {
			gCurrentEditorState.currentImgEmbedding = {
				data: data.embeddings,
				dims: data.dims
			};
			gIsEncoderRunning = false;
		} else if (type === MESSAGE_TYPES.INPAINTING_RUN_RESULT_SUCCESS && !gInpainterLoading) {
			RGB_CHW_array_to_imageData(data, gImageCanvas.height, gImageCanvas.width).then(
				(resultImgData) => {
					handleInpaintedImgData(resultImgData);
					gInpaintingRunning = false;
				}
			);
		} else if (type === MESSAGE_TYPES.ALL_MODELS_LOADED) {
			gDecoderLoading = gEncoderLoading = gInpainterLoading = false;
			if (
				gCurrentEditorState &&
				gCurrentEditorState.imgData &&
				!gIsEncoderRunning &&
				!gCurrentEditorState.currentImgEmbedding
			) {
				gIsEncoderRunning = true;
				runEncoderCurrentState();
			}
		} else if (type === MESSAGE_TYPES.ENCODER_DECODER_LOADED) {
			gEncoderLoading = gDecoderLoading = false;
			if (
				gCurrentEditorState &&
				gCurrentEditorState.imgData &&
				!gIsEncoderRunning &&
				!gCurrentEditorState.currentImgEmbedding
			) {
				gIsEncoderRunning = true;
				runEncoderCurrentState().then((_) => {
					$mainWorker?.postMessage({ type: MESSAGE_TYPES.LOAD_INPAINTER });
				});
			}
		}
	}

	onMount(async () => {
		let header = document.querySelector('header');
		if (header) {
			gHeaderHeightPx = header.getBoundingClientRect().height;
		}
		if ($uploadedImgBase64 === null || $uploadedImgFileName === '') {
			goto(base == '' ? '/' : base);
		}
		gUploadedImage = $uploadedImgBase64;
		window.addEventListener('resize', () => {
			if (gImageCanvas) {
				const canvasElementSize = gImageCanvas.getBoundingClientRect();
				gImgResToCanvasSizeRatio = gImageCanvas.width / canvasElementSize.width;
			}
		});
		if (!gUploadedImage || !uploadedImgFileName) {
			goto(base == '' ? '/' : base);
			return;
		}
		gCurrentEditorState = await initEditorState(gUploadedImage, $uploadedImgFileName);

		if (!$mainWorker) {
			goto(base == '' ? '/' : base);
		} else {
			/*mainworker has set modelsLoaded function callback (setups editor etc.)
			it is called either based on response to this message or after models are loaded 
			(then, the message is sent automatically after loading)*/
			$mainWorker.postMessage({ type: MESSAGE_TYPES.CHECK_MODELS_LOADING_STATE });
		}

		gPanzoomObj = Panzoom(gCanvasesContainer, {
			disablePan: !gPanEnabled,
			minScale: 1,
			maxScale: 10,
			disableZoom: gAnythingEssentialLoading,
			cursor: 'default'
		}) as PanzoomObject;
		gCanvasesContainer.parentElement!.addEventListener('wheel', gPanzoomObj.zoomWithWheel);
		gCanvasesContainer.addEventListener('panzoomchange', (event: any) => {
			gCurrentZoom = event.detail.scale;
		});
		setTimeout(() => gPanzoomObj.zoom(0, 100));
		gPanzoomObj.zoom(5, { animate: true });
	});

	//change dilatation when pixelsDilatation value changes
	async function changeDilatation(pixelsDilatation: number, currentEditorState: editorState) {
		if (currentEditorState) {
			currentEditorState.maskSAMDilated = await dilateMaskByPixels(
				pixelsDilatation,
				currentEditorState.maskSAM
			);
			renderEditorState(currentEditorState, gImageCanvas, gMaskCanvas, gImgResToCanvasSizeRatio);
		}
	}

	//initializes editor state on new image
	const initEditorState = async (sourceImgBase64Data: string, sourceImgName: string) => {
		return new Promise<editorState>((resolve, reject) => {
			const img = new Image();
			img.src = sourceImgBase64Data;
			gImgName = sourceImgName;
			img.onload = async () => {
				// Calculate aspect ratio
				gImageCanvas.width = img.width;
				gMaskCanvas.width = img.width;
				gImageCanvas.height = img.height;
				gMaskCanvas.height = img.height;
				const canvasElementSize = gImageCanvas.getBoundingClientRect();
				gImgResToCanvasSizeRatio = img.width / canvasElementSize.width;

				//render image
				const ctx = gImageCanvas.getContext('2d');
				clearCanvas(gImageCanvas);
				ctx.drawImage(
					img,
					0,
					0,
					img.width,
					img.height,
					0,
					0,
					gImageCanvas.width,
					gImageCanvas.height
				);

				gImgDataOriginal = getImageData(gImageCanvas);
				//setup editor state
				gEditorStatesHistory = [];
				gEditorStatesUndoed = [];
				let newEditorState = {
					maskBrush: createEmptyMaskArray(gImgDataOriginal.width, gImgDataOriginal.height),
					maskSAM: createEmptyMaskArray(gImgDataOriginal.width, gImgDataOriginal.height),
					maskSAMDilated: createEmptyMaskArray(gImgDataOriginal.width, gImgDataOriginal.height),
					clickedPositions: new Array<SAMmarker>(),
					imgData: gImgDataOriginal,
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
		if (!gCurrentEditorState.currentImgEmbedding) {
			return;
		}
		gDecoderRunning = true;
		const modelInput = await createDecoderInputDict(
			gCurrentEditorState,
			gResizedImgWidth,
			gResizedImgHeight
		);
		$mainWorker?.postMessage({
			type: MESSAGE_TYPES.DECODER_RUN,
			data: modelInput
		});
	};

	async function handleCanvasClick(event: MouseEvent) {
		console.log('canvas log');
		console.log(gImageCanvas.getBoundingClientRect().width);
		if (gAnythingEssentialLoading) {
			return;
		}
		//it logs -0 at 0,0 for some reason
		event.preventDefault();
		const xScaled = Math.abs(event.offsetX) * gImgResToCanvasSizeRatio;
		const yScaled = Math.abs(event.offsetY) * gImgResToCanvasSizeRatio;

		console.log(event.offsetX, event.offsetY);
		console.log('scaled');
		console.log(xScaled, yScaled);

		gEditorStatesHistory = [...gEditorStatesHistory, gCurrentEditorState];
		gCurrentEditorState = {
			maskBrush: gCurrentEditorState.maskBrush,
			maskSAM: gCurrentEditorState.maskSAM,
			maskSAMDilated: gCurrentEditorState.maskSAMDilated,
			clickedPositions: new Array<SAMmarker>(...gCurrentEditorState.clickedPositions, {
				x: xScaled,
				y: yScaled,
				// type: event.button === 0 ? 'positive' : 'negative'
				type: gSelectedSAMMode
			}),
			imgData: gCurrentEditorState.imgData,
			currentImgEmbedding: gCurrentEditorState.currentImgEmbedding
		} as editorState;
		renderEditorState(gCurrentEditorState, gImageCanvas, gMaskCanvas, gImgResToCanvasSizeRatio);
		runDecoderCurrentState();
	}

	function undoLastEditorAction() {
		//new state management
		if (gEditorStatesHistory.length > 0 && gCurrentEditorState) {
			gEditorStatesUndoed = [...gEditorStatesUndoed, gCurrentEditorState];
			gCurrentEditorState = gEditorStatesHistory[gEditorStatesHistory.length - 1];
			gEditorStatesHistory = gEditorStatesHistory.slice(0, -1);
			renderEditorState(gCurrentEditorState, gImageCanvas, gMaskCanvas, gImgResToCanvasSizeRatio);
		}
	}

	function redoLastEditorAction() {
		//new state management
		if (gEditorStatesUndoed.length > 0 && gCurrentEditorState) {
			gEditorStatesHistory = [...gEditorStatesHistory, gCurrentEditorState];
			gCurrentEditorState = gEditorStatesUndoed[gEditorStatesUndoed.length - 1];
			gEditorStatesUndoed = gEditorStatesUndoed.slice(0, -1);
			renderEditorState(gCurrentEditorState, gImageCanvas, gMaskCanvas, gImgResToCanvasSizeRatio);
		}
	}

	function resetEditorState() {
		//clear all states
		gPanzoomObj.reset();
		gCurrentEditorState = {
			maskBrush: createEmptyMaskArray(gImgDataOriginal.width, gImgDataOriginal.height),
			maskSAM: createEmptyMaskArray(gImgDataOriginal.width, gImgDataOriginal.height),
			maskSAMDilated: createEmptyMaskArray(gImgDataOriginal.width, gImgDataOriginal.height),
			clickedPositions: [],
			imgData: gImgDataOriginal,
			currentImgEmbedding: gEditorStatesHistory[0]
				? gEditorStatesHistory[0].currentImgEmbedding
				: gCurrentEditorState.currentImgEmbedding
		};
		gEditorStatesHistory = [];
		gEditorStatesUndoed = [];
		renderEditorState(gCurrentEditorState, gImageCanvas, gMaskCanvas, gImgResToCanvasSizeRatio);
	}

	// brush tool
	function startPaintingMouse(event: MouseEvent, canvas: HTMLCanvasElement) {
		gIsPainting = true;
		gPrevMouseX = event.offsetX * gImgResToCanvasSizeRatio;
		gPrevMouseY = event.offsetY * gImgResToCanvasSizeRatio;
		handleEditorCursorMove(event, canvas);
	}
	function startPaintingTouch(event: TouchEvent, canvas: HTMLCanvasElement) {
		gIsPainting = true;

		let touch = event.touches[0]; // Get the first touch, you might handle multi-touch differently
		let targetRect = canvas.getBoundingClientRect(); // Get the target element's position

		// Calculate offsetX and offsetY
		let offsetX = touch.clientX - targetRect.left;
		let offsetY = touch.clientY - targetRect.top;

		gPrevMouseX = offsetX * gImgResToCanvasSizeRatio;
		gPrevMouseY = offsetY * gImgResToCanvasSizeRatio;
	}

	function stopPainting() {
		gIsPainting = false;
		const ctx = gMaskCanvas.getContext('2d', { willReadFrequently: true });
		if (!ctx) {
			console.error('canvas context error');
			return;
		}
		const imageData = ctx.getImageData(0, 0, gMaskCanvas.width, gMaskCanvas.height);
		let maskArray: boolean[][] = maskArrayFromImgData(
			imageData,
			gMaskCanvas.width,
			gMaskCanvas.height
		);
		//new state management
		gEditorStatesHistory = [...gEditorStatesHistory, gCurrentEditorState];
		gCurrentEditorState = {
			maskBrush: maskArray,
			maskSAM: gCurrentEditorState.maskSAM,
			maskSAMDilated: gCurrentEditorState.maskSAMDilated,
			clickedPositions: gCurrentEditorState.clickedPositions,
			imgData: gCurrentEditorState.imgData,
			currentImgEmbedding: gCurrentEditorState.currentImgEmbedding
		} as editorState;
		gEditorStatesUndoed = [];
	}
	function handleEditorCursorMove(event: MouseEvent | TouchEvent, canvas: HTMLCanvasElement) {
		let x: number, y: number;
		if (event instanceof MouseEvent) {
			x = event.offsetX * gImgResToCanvasSizeRatio;
			y = event.offsetY * gImgResToCanvasSizeRatio;
			gCurrentCanvasRelativeX = event.offsetX;
			gCurrentCanvasRelativeY = event.offsetY;
		} else {
			let touch = event.touches[0]; // Get the first touch, you might handle multi-touch differently
			let targetRect = canvas.getBoundingClientRect(); // Get the target element's position

			// Calculate offsetX and offsetY
			let offsetX = touch.clientX - targetRect.left;
			let offsetY = touch.clientY - targetRect.top;
			x = offsetX * gImgResToCanvasSizeRatio;
			y = offsetY * gImgResToCanvasSizeRatio;
			gCurrentCanvasRelativeX = offsetX;
			gCurrentCanvasRelativeY = offsetY;
		}
		if (gIsPainting) {
			let canvasCtx = canvas.getContext('2d', { willReadFrequently: true });
			if (!canvasCtx) {
				console.error('canvas context error');
				return;
			}
			const { currentX, currentY } = canvasBrushDraw(
				x,
				y,
				gPrevMouseX,
				gPrevMouseY,
				gBrushToolSize,
				canvasCtx,
				gImgResToCanvasSizeRatio
			);
			gPrevMouseX = currentX;
			gPrevMouseY = currentY;
		}
	}

	function getImageData(canvas: HTMLCanvasElement) {
		return canvas.getContext('2d')!.getImageData(0, 0, canvas.width, canvas.height);
	}

	function handleBrushModeChange(brushMode: 'brush' | 'eraser', maskCanvas: HTMLCanvasElement) {
		// Get the canvas element
		if (!maskCanvas || !brushMode) return;
		const context = maskCanvas.getContext('2d');
		if (!context) return;

		context.globalCompositeOperation = brushMode === 'brush' ? 'source-over' : 'destination-out';
	}

	async function handleInpainting() {
		if (!gAnythingEssentialLoading) {
			gInpaintingRunning = true;
			runInpainting(gCurrentEditorState, $mainWorker!);
		}
	}

	//handle inpainted image data - set and render new editor state and run encoder on new image
	async function handleInpaintedImgData(inpaintedImgData: ImageData) {
		gEditorStatesHistory = [...gEditorStatesHistory, gCurrentEditorState];
		let newEditorState: editorState = {
			maskBrush: createEmptyMaskArray(gImgDataOriginal.width, gImgDataOriginal.height),
			maskSAM: createEmptyMaskArray(gImgDataOriginal.width, gImgDataOriginal.height),
			maskSAMDilated: createEmptyMaskArray(gImgDataOriginal.width, gImgDataOriginal.height),
			clickedPositions: [],
			imgData: inpaintedImgData,
			currentImgEmbedding: undefined
		};
		gCurrentEditorState = newEditorState;
		renderEditorState(gCurrentEditorState, gImageCanvas, gMaskCanvas, gImgResToCanvasSizeRatio);
		gIsEncoderRunning = true;
		runEncoderCurrentState();
	}

	function handlePanzoomSettingsChange(enablePan: boolean, anythingEssentialLoading: boolean) {
		if (gPanzoomObj) {
			gPanzoomObj.setOptions({
				disablePan: !enablePan,
				disableZoom: anythingEssentialLoading
			});
		}
	}

</script>

<Drawer>
	<h2 class="p-4 font-semibold">Smart object remover</h2>
	<hr />
	<EditorToolSelection
		bind:selectedTool={gSelectedTool}
		bind:selectedBrushMode={gSelectedBrushMode}
		bind:selectedSAMMode={gSelectedSAMMode}
		bind:SAMMaskDilatation={gSAMMaskDilatation}
		bind:brushToolSize={gBrushToolSize}
	/>
</Drawer>

<AppShell slotSidebarLeft="overflow-visible lg:w-80 w-0 h-screen shadow-md z-50">
	<svelte:fragment slot="header">
		<Navbar
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
		<EditorToolSelection
			bind:selectedTool={gSelectedTool}
			bind:selectedBrushMode={gSelectedBrushMode}
			bind:selectedSAMMode={gSelectedSAMMode}
			bind:SAMMaskDilatation={gSAMMaskDilatation}
			bind:brushToolSize={gBrushToolSize}
		/>
	</svelte:fragment>
	<div
		class="
	flex flex-col gap-y-4
	2xl:px-64 xl:px-16 md:px-8 px-2 py-4
	"
		style="max-height: calc(100vh - {gHeaderHeightPx}px)"
	>
		<!-- top buttons panel -->
		<div class="flex flex-none justify-between">
			<!-- left buttons -->
			<div class="flex lg:gap-x-2 gap-x-1">
				<button
					class="btn btn-sm lg:btn-md variant-filled"
					on:click={undoLastEditorAction}
					disabled={gEditorStatesHistory.length === 0 || gAnythingEssentialLoading}
					><Undo class="lg:w-6 lg:h-6 w-4 h-4" /></button
				>
				<button
					class="btn btn-sm lg:btn-md variant-filled"
					on:click={redoLastEditorAction}
					disabled={gEditorStatesUndoed.length === 0 || gAnythingEssentialLoading}
					><Redo class="lg:w-6 lg:h-6 w-4 h-4" /></button
				>
				<button
					disabled={gAnythingEssentialLoading}
					class="btn btn-sm lg:btn-md variant-filled"
					on:click={resetEditorState}><RotateCw class="lg:w-6 lg:h-6 w-4 h-4" /></button
				>
			</div>

			<!-- right buttons -->
			<div class="flex lg:gap-x-2 gap-x-1">
				<button
					disabled={gAnythingEssentialLoading}
					class="btn btn-sm lg:btn-md variant-filled select-none"
					on:mousedown={() => {
						gShowOriginalImage = true;
					}}
					on:touchstart={() => {
						gShowOriginalImage = true;
					}}
					on:mouseup={() => {
						gShowOriginalImage = false;
					}}
					on:touchend={() => {
						gShowOriginalImage = false;
					}}
				>
					<span class="hidden sm:inline no-margin">Hold to compare</span>
					<span class="no-margin sm:hidden">Compare</span>
				</button>

				<button
					disabled={gAnythingEssentialLoading}
					class="btn btn-sm lg:btn-md variant-filled"
					on:click={() => downloadImage(gCurrentEditorState.imgData, gImgName)}
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
			style="cursor: {gAnythingEssentialLoading ? 'not-allowed' : 'default'}"
		>
			<div
				class="absolute !w-full !h-full flex items-center flex-col gap-y-2 justify-center z-10 {!gAnythingEssentialLoading
					? 'hidden'
					: ''}"
			>
				<ProgressRadial meter="stroke-primary-500" track="stroke-primary-500/30" />
				<div class="text-primary-500 font-bold text-2xl mt-2">
					{#if gEncoderLoading || gDecoderLoading}
						Models loading...
					{:else if gInpaintingRunning}
						Removing area...
					{:else if gIsEncoderRunning}
						Processing new image...
					{:else if gDecoderRunning}
						Computing mask...
					{/if}
				</div>
			</div>
			<div class="canvases w-full" bind:this={gCanvasesContainer}>
				<div class="relative !h-full !w-full flex justify-center" role="group">
					<div class="flex-none" />
					<!-- default width so the page isnt empty till load -->
					<div
						class="relative flex-none shrink"
						style="cursor: {currentCanvasCursor(gPanEnabled, gSelectedTool)}"
						on:mouseenter={() => {
							gDisplayBrushCursor =
								gSelectedTool === 'brush' && !gAnythingEssentialLoading && !gPanEnabled;
						}}
						on:mouseleave={() => {
							gDisplayBrushCursor = false;
						}}
						role="group"
					>
						<div class="absolute w-full h-full overflow-hidden">
							<div
								id="brushToolCursor"
								role="group"
								class={gDisplayBrushCursor ? 'block' : 'hidden'}
								style="
			width: {gBrushToolSize - 2}px;
			height: {gBrushToolSize - 2}px;
			left: {gCurrentCanvasRelativeX}px;
			top: {gCurrentCanvasRelativeY}px;
			background-color: {gSelectedBrushMode === 'brush' ? '#408dff' : '#f5f5f5'};
			border: 1px solid {gSelectedBrushMode === 'brush' ? '#0261ed' : '#bfbfbf'};
			opacity: {gIsPainting ? 0.6 : 0.5};
		"
							/>
						</div>
						<canvas
							class="shadow-lg
						{gAnythingEssentialLoading ? 'opacity-30 cursor-not-allowed' : ''} 
						{gShowOriginalImage === true ? '!hidden' : '!block'}
						
						"
							id="imageCanvas"
							bind:this={gImageCanvas}
						/>

						<canvas
							id="maskCanvas"
							class="
						{gPanEnabled ? '' : 'panzoom-exclude'} 
						{gAnythingEssentialLoading ? 'opacity-30 cursor-not-allowed' : ''}
						{gShowOriginalImage ? '!hidden' : '!block'}
						"
							bind:this={gMaskCanvas}
							on:mousedown={gSelectedTool === 'brush' && !gPanEnabled
								? (e) => startPaintingMouse(e, gMaskCanvas)
								: undefined}
							on:mouseup={gSelectedTool === 'brush' && !gPanEnabled ? stopPainting : undefined}
							on:mousemove={(event) =>
								gSelectedTool === 'brush' && !gPanEnabled
									? handleEditorCursorMove(event, gMaskCanvas)
									: undefined}
							on:touchstart={gSelectedTool === 'brush'
								? // && !enablePan
								  (e) => {
										console.log('start');
										console.log(e);
										if (e.touches.length === 1) {
											startPaintingTouch(e, gMaskCanvas);
										} else {
											stopPainting();
										}
								  }
								: undefined}
							on:touchend={gSelectedTool === 'brush'
								? // &&  !enablePan
								  (e) => {
										console.log('end');
										console.log(e);
										stopPainting();
								  }
								: undefined}
							on:touchmove={gSelectedTool === 'brush'
								? // &&  !enablePan
								  (e) => {
										if (e.touches.length === 1) {
											e.preventDefault();
											handleEditorCursorMove(e, gMaskCanvas);
										}
								  }
								: undefined}
							on:click={gSelectedTool === 'segment_anything' && !gPanEnabled
								? async (e) => {
										console.log(e);
										handleCanvasClick(e);
								  }
								: undefined}
						/>

						<img
							class="shadow-lg
						{gAnythingEssentialLoading ? 'opacity-50 cursor-not-allowed' : ''}
						{gShowOriginalImage === true ? '!block' : '!hidden'}

						"
							src={$uploadedImgBase64}
							alt="originalImage"
							bind:this={gOriginalImgElement}
						/>
					</div>
					<div class="flex-none" />
				</div>
			</div>
		</div>
		<!-- bottom buttons -->
		<div class="flex flex-none flex-wrap lg:gap-x-2 gap-x-1">
			<div class="sm:flex-1 flex-0" />
			<div>
				<PanzoomCanvasControls
					disabled={gAnythingEssentialLoading}
					currentZoomValue={gCurrentZoom}
					bind:panEnabled={gPanEnabled}
					on:zoomIn={() => gPanzoomObj.zoomIn()}
					on:zoomOut={() => gPanzoomObj.zoomOut()}
					on:zoomReset={() => gPanzoomObj.reset()}
				/>
			</div>
			<div class="flex-1 flex justify-end">
				<button
					class="btn lg:btn-xl md:btn-md btn-sm variant-filled-primary text-white dark:text-white font-semibold"
					disabled={gAnythingEssentialLoading || gInpainterLoading}
					on:click={async () => handleInpainting()}
				>
					<span class="flex gap-x-2 items-center">
						{#if gInpainterLoading && !gAnythingEssentialLoading}
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
