<script lang="ts">
	import { onMount } from 'svelte';
	import { base } from '$app/paths';
	import { goto } from '$app/navigation';
	import Panzoom, { type PanzoomObject } from '@panzoom/panzoom';
	import { WandSparkles, Undo, Redo, RotateCw, HardDriveDownload } from 'lucide-svelte';
	import { AppShell, Drawer, getDrawerStore, ProgressRadial } from '@skeletonlabs/skeleton';
	import { mainWorker } from '../../stores/workerStore';
	import { MESSAGE_TYPES } from '../../workers/messageTypes';
	import { uploadedImgBase64, uploadedImgFileName } from '../../stores/imgStore';
	import Navbar from './../../components/Navbar.svelte';
	import EditorToolSelection from './../../components/EditorToolSelection.svelte';
	import PanzoomCanvasControls from '../../components/PanzoomCanvasControls.svelte';
	import type { tool, brushMode, SAMMode } from '../../types/editorTypes';
	import {
		renderEditorState,
		type editorState,
		type SAMmarker,
		canvasBrushDraw,
		runInpainting,
		createEmptyMaskArray,
		createDecoderInputDict,
		runModelEncoder,
		RGB_CHW_array_to_imageData,
		currentCanvasCursor,
		decoderResultToMaskArray
	} from './editorModule';
	import { getResizedImgData } from '$lib/onnxHelpers';
	import { dilateMaskByPixels, maskArrayFromImgData, downloadImage } from '$lib/editorHelpers';

	//models loading/processing state globals
	let gEncoderLoading = true;
	let gDecoderLoading = true;
	let gInpainterLoading = true;
	let gIsEncoderRunning = false;
	let gDilatationProcessing = false;
	let gDecoderRunning = false;
	let gInpaintingRunning = false;
	let gIsPainting = false;

	//general editor state globals
	let gImgDataOriginal: ImageData;
	let gImgName: string = 'default';
	let gImgResToCanvasSizeRatio: number = 1;
	let gCurrentEditorState: editorState;
	let gEditorStatesHistory: editorState[] = [];
	let gEditorStatesUndoed: editorState[] = [];

	//element references/UI globals
	let gImageCanvas: any;
	let gCanvasesContainer: any;
	let gMaskCanvas: any;
	let gOriginalImgElement: HTMLImageElement;
	let gHeaderHeightPx = 0;
	let gShowOriginalImage: boolean = false;
	const drawerStore = getDrawerStore();

	//SAM globals
	let gResizedImgWidth: number;
	let gResizedImgHeight: number;

	//canvas brush drawing global
	let gPrevMouseX = 0;
	let gPrevMouseY = 0;
	let gBrushOffsetX = 0;
	let gBrushOffsetY = 0;
	let gDisplayBrushCursor: boolean = false;

	//sidebar controls globals
	let gSAMMaskDilatation: number = 7;
	$: gSAMMaskDilatationResScaled = gImageCanvas ? gSAMMaskDilatation * (Math.max(gImageCanvas.width, gImageCanvas.height) / 1024) : 1;
	let gSelectedBrushMode: brushMode; // Initial selected option
	let gSelectedTool: tool;
	let gSelectedSAMMode: SAMMode;
	let gBrushToolSize: number;
	let gDilatationDebounceTimeout: NodeJS.Timeout;

	//canvas panzoom globals
	let gPanEnabled: boolean = false;
	let gCurrentZoom: number = 1;
	let gPanzoomObj: PanzoomObject;

	$: handleBrushModeChange(gSelectedBrushMode, gMaskCanvas);
	//scale mask dilatation by resolution, so the mask isnt too big on low res and too small on high res
	$: handleDilatationChange(gSAMMaskDilatationResScaled, 100)

	$: handlePanzoomSettingsChange(gPanEnabled, gAnythingEssentialLoading);
	$: gAnythingEssentialLoading =
		gInpaintingRunning ||
		gEncoderLoading ||
		gDecoderLoading ||
		gIsEncoderRunning ||
		gDecoderRunning ||
		gDilatationProcessing;

	if ($mainWorker) {
		$mainWorker.onmessage = handleWorkerMessages;
	}

	//resize img to longside 1024 and run encoder
	const runEncoderCurrentState = async () => {
		const SAMLongside = 1024;
		getResizedImgData(gCurrentEditorState.imgData, SAMLongside).then((resizedImgRGBData) => {
			//set globals for decoder (needs to map input points to correct coordinates in resized image)
			gResizedImgWidth = resizedImgRGBData.width;
			gResizedImgHeight = resizedImgRGBData.height;
			runModelEncoder(resizedImgRGBData, $mainWorker!);
		});
	};

	//webworker incoming messages handling
	function handleWorkerMessages(event: MessageEvent<any>) {
		const { data, type } = event.data;
		if (type === MESSAGE_TYPES.INPAINTER_LOADED) {
			gInpainterLoading = false;
		}
		if (type === MESSAGE_TYPES.DECODER_RUN_RESULT_SUCCESS && !gDecoderLoading) {
			//map decoder result to 2D bool mask, dilate it and render new editor state
			const SAMMaskArray: boolean[][] = decoderResultToMaskArray(data);
			if (SAMMaskArray) {
				dilateMaskByPixels(gSAMMaskDilatationResScaled, SAMMaskArray).then((dilatedMask) => {
					gCurrentEditorState.maskSAMDilated = dilatedMask;
					gCurrentEditorState.maskSAM = SAMMaskArray;
					renderEditorState(
						gCurrentEditorState,
						gImageCanvas,
						gMaskCanvas,
						gImgResToCanvasSizeRatio
					);
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
			//convert CHW result array to imgData type and set new editor state + run encoder
			RGB_CHW_array_to_imageData(data, gImageCanvas.height, gImageCanvas.width).then(
				(resultImgData) => {
					setInpaintedImgEditorState(resultImgData);
					gInpaintingRunning = false;
					gIsEncoderRunning = true;
					runEncoderCurrentState();
				}
			);
		} else if (type === MESSAGE_TYPES.ALL_MODELS_LOADED) {
			//all models loaded - set globals and run encoder if img is ready
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
			//SAM loaded, run encoder if img is ready and load inpainter model then
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

	//init editor state on editor page load
	onMount(async () => {
		//for computing height limit of editor canvases
		let header = document.querySelector('header');
		if (header) {
			gHeaderHeightPx = header.getBoundingClientRect().height;
		}
		//change ratio of canvas and real img res on win resize
		window.addEventListener('resize', () => {
			if (gImageCanvas) {
				const canvasElementSize = gImageCanvas.getBoundingClientRect();
				gImgResToCanvasSizeRatio = gImageCanvas.width / canvasElementSize.width;
			}
		});
		//if any of these is null, something went wrong, go back to homepage
		if ($uploadedImgBase64 === null || $uploadedImgFileName === '' || !$mainWorker) {
			goto(base == '' ? '/' : base);
			return;
		}
		//init editor and control elements state
		gCurrentEditorState = await initEditorState($uploadedImgBase64, $uploadedImgFileName);
		gPanzoomObj = Panzoom(gCanvasesContainer, {
			disablePan: !gPanEnabled,
			minScale: 1,
			maxScale: 10,
			disableZoom: gAnythingEssentialLoading,
			cursor: 'default'
		}) as PanzoomObject;
		//add event listeners
		gCanvasesContainer.parentElement!.addEventListener('wheel', gPanzoomObj.zoomWithWheel);
		gCanvasesContainer.addEventListener('panzoomchange', (event: any) => {
			gCurrentZoom = event.detail.scale;
		});
		//after all inits, check if models are loaded
		$mainWorker.postMessage({ type: MESSAGE_TYPES.CHECK_MODELS_LOADING_STATE });
	});

	//change dilatation when pixelsDilatation value changes
	async function handleDilatationChange(pixelsDilatation: number, debouncems = 100) {
		clearTimeout(gDilatationDebounceTimeout);
		gDilatationProcessing = true
		gDilatationDebounceTimeout = setTimeout(async () => {
			if (gCurrentEditorState) {
				gCurrentEditorState.maskSAMDilated = await dilateMaskByPixels(
					pixelsDilatation,
					gCurrentEditorState.maskSAM
				);
				renderEditorState(gCurrentEditorState, gImageCanvas, gMaskCanvas, gImgResToCanvasSizeRatio).then(() => {
					gDilatationProcessing = false
				});
			}
		}, debouncems);
	}

	//initializes editor state on new image
	const initEditorState = async (sourceImgBase64Data: string, sourceImgName: string) => {
		return new Promise<editorState>((resolve, reject) => {
			gImgName = sourceImgName;
			const img = new Image();
			img.src = sourceImgBase64Data;
			img.onload = async () => {
				// Calculate aspect ratio
				gImageCanvas.width = gMaskCanvas.width = img.width;
				gImageCanvas.height = gMaskCanvas.height = img.height;
				gImgResToCanvasSizeRatio = img.width / gImageCanvas.getBoundingClientRect().width;
				console.log(Math.max(gImageCanvas.width, gImageCanvas.height) / 1024)
				//render image
				const ctx = gImageCanvas.getContext('2d');
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
				resolve({
					maskBrush: createEmptyMaskArray(gImgDataOriginal.width, gImgDataOriginal.height),
					maskSAM: createEmptyMaskArray(gImgDataOriginal.width, gImgDataOriginal.height),
					maskSAMDilated: createEmptyMaskArray(gImgDataOriginal.width, gImgDataOriginal.height),
					clickedPositions: new Array<SAMmarker>(),
					imgData: gImgDataOriginal,
					currentImgEmbedding: undefined
				} as editorState);
			};

			img.onerror = (error) => {
				reject(error);
			};
		});
	};

	//post message with decoder input dict to webworker
	async function handleCanvasClick(event: MouseEvent) {
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

		if (gAnythingEssentialLoading) {
			return;
		}
		event.preventDefault();

		gEditorStatesHistory = [...gEditorStatesHistory, gCurrentEditorState];
		gCurrentEditorState = {
			maskBrush: gCurrentEditorState.maskBrush,
			maskSAM: gCurrentEditorState.maskSAM,
			maskSAMDilated: gCurrentEditorState.maskSAMDilated,
			clickedPositions: new Array<SAMmarker>(...gCurrentEditorState.clickedPositions, {
				x: Math.abs(event.offsetX) * gImgResToCanvasSizeRatio, // for [0,0] it is sometimes -0
				y: Math.abs(event.offsetY) * gImgResToCanvasSizeRatio,
				type: gSelectedSAMMode
			}),
			imgData: gCurrentEditorState.imgData,
			currentImgEmbedding: gCurrentEditorState.currentImgEmbedding
		};
		renderEditorState(gCurrentEditorState, gImageCanvas, gMaskCanvas, gImgResToCanvasSizeRatio);
		runDecoderCurrentState();
	}

	//undo
	function undoLastEditorAction() {
		if (gEditorStatesHistory.length > 0 && gCurrentEditorState) {
			gEditorStatesUndoed = [...gEditorStatesUndoed, gCurrentEditorState];
			gCurrentEditorState = gEditorStatesHistory[gEditorStatesHistory.length - 1];
			gEditorStatesHistory = gEditorStatesHistory.slice(0, -1);
			renderEditorState(gCurrentEditorState, gImageCanvas, gMaskCanvas, gImgResToCanvasSizeRatio);
		}
	}
	//redo
	function redoLastEditorAction() {
		if (gEditorStatesUndoed.length > 0 && gCurrentEditorState) {
			gEditorStatesHistory = [...gEditorStatesHistory, gCurrentEditorState];
			gCurrentEditorState = gEditorStatesUndoed[gEditorStatesUndoed.length - 1];
			gEditorStatesUndoed = gEditorStatesUndoed.slice(0, -1);
			renderEditorState(gCurrentEditorState, gImageCanvas, gMaskCanvas, gImgResToCanvasSizeRatio);
		}
	}

	//reset
	function resetEditorState() {
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

	//init painting globals, draw first brush stroke
	function startPainting(event: MouseEvent | TouchEvent, canvas: HTMLCanvasElement) {
		gIsPainting = true;
		if (event instanceof MouseEvent) {
			gPrevMouseX = event.offsetX * gImgResToCanvasSizeRatio;
			gPrevMouseY = event.offsetY * gImgResToCanvasSizeRatio;
		} else {
			let touch = event.touches[0]; // Get the first touch, you might handle multi-touch differently
			gPrevMouseX =
				(touch.clientX - canvas.getBoundingClientRect().left) * gImgResToCanvasSizeRatio;
			gPrevMouseY = (touch.clientY - canvas.getBoundingClientRect().top) * gImgResToCanvasSizeRatio;
		}
		handleEditorCursorMove(event, canvas);
	}

	//stop painting and save mask array to editor state
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
		gCurrentEditorState = { ...gCurrentEditorState, maskBrush: maskArray };
		gEditorStatesUndoed = [];
	}

	//draw brush stroke on canvas
	function handleEditorCursorMove(event: MouseEvent | TouchEvent, canvas: HTMLCanvasElement) {
		let x: number, y: number;
		if (event instanceof MouseEvent) {
			x = event.offsetX * gImgResToCanvasSizeRatio;
			y = event.offsetY * gImgResToCanvasSizeRatio;
			gBrushOffsetX = event.offsetX;
			gBrushOffsetY = event.offsetY;
		} else {
			let touch = event.touches[0]; // Get the first touch, you might handle multi-touch differently
			let targetRect = canvas.getBoundingClientRect(); // Get the target element's position
			// Calculate offsetX and offsetY
			let offsetX = touch.clientX - targetRect.left;
			let offsetY = touch.clientY - targetRect.top;
			x = offsetX * gImgResToCanvasSizeRatio;
			y = offsetY * gImgResToCanvasSizeRatio;
			gBrushOffsetX = offsetX;
			gBrushOffsetY = offsetY;
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

	//get canvas image data
	function getImageData(canvas: HTMLCanvasElement) {
		return canvas.getContext('2d')!.getImageData(0, 0, canvas.width, canvas.height);
	}

	//change brush mode between brush and eraser - destionation-out composite removes pixels from mask canvas
	function handleBrushModeChange(brushMode: 'brush' | 'eraser', maskCanvas: HTMLCanvasElement) {
		// Get the canvas element
		if (!maskCanvas || !brushMode) return;
		const context = maskCanvas.getContext('2d');
		if (!context) return;

		context.globalCompositeOperation = brushMode === 'brush' ? 'source-over' : 'destination-out';
	}

	//run inpainting on current editor state
	async function handleInpainting() {
		if (!gAnythingEssentialLoading) {
			gInpaintingRunning = true;
			runInpainting(gCurrentEditorState, $mainWorker!);
		}
	}

	//handle inpainted image data - set and render new editor state and run encoder on new image
	async function setInpaintedImgEditorState(inpaintedImgData: ImageData) {
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
		return renderEditorState(
			gCurrentEditorState,
			gImageCanvas,
			gMaskCanvas,
			gImgResToCanvasSizeRatio
		);
	}

	//set pan and zoom enabled/disabled state
	function handlePanzoomSettingsChange(enablePan: boolean, anythingEssentialLoading: boolean) {
		if (gPanzoomObj) {
			gPanzoomObj.setOptions({
				disablePan: !enablePan,
				disableZoom: anythingEssentialLoading
			});
		}
	}
</script>

<!-- drawer menu on phone -->
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
			links={[{ name: 'Home', href: base == '' ? '/' : base }]}
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
		class="flex flex-col gap-y-4 2xl:px-64 xl:px-16 md:px-8 px-2 py-4"
		style="max-height: calc(100vh - {gHeaderHeightPx}px)"
	>
		<!-- upper buttons rows -->
		<div class="flex flex-none justify-between">
			<div class="flex lg:gap-x-2 gap-x-1">
				<button
					class="btn btn-sm lg:btn-md variant-filled"
					on:click={undoLastEditorAction}
					disabled={gEditorStatesHistory.length === 0 || gAnythingEssentialLoading}
				>
					<Undo class="lg:w-6 lg:h-6 w-4 h-4" />
				</button>
				<button
					class="btn btn-sm lg:btn-md variant-filled"
					on:click={redoLastEditorAction}
					disabled={gEditorStatesUndoed.length === 0 || gAnythingEssentialLoading}
				>
					<Redo class="lg:w-6 lg:h-6 w-4 h-4" />
				</button>
				<button
					disabled={gAnythingEssentialLoading}
					class="btn btn-sm lg:btn-md variant-filled"
					on:click={resetEditorState}
				>
					<RotateCw class="lg:w-6 lg:h-6 w-4 h-4" />
				</button>
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
					<span class="hidden sm:inline inset-0">Hold to compare</span>
					<span class="sm:hidden inset-0">Compare</span>
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
		<!-- canvases -->
		<div
			id="mainEditorContainer"
			class="grow overflow-hidden relative flex justify-center"
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
					{:else if gDecoderRunning || gDilatationProcessing}
						Computing mask...
					{:else}
						Loading...
					{/if}
				</div>
			</div>
			<div class="canvases w-full" bind:this={gCanvasesContainer}>
				<div class="relative !h-full !w-full flex justify-center" role="group">
					<div class="flex-none" />
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
								class="
								overflow-hidden absolute rounded-full pointer-events-none z-50
								{gDisplayBrushCursor ? 'block' : 'hidden'}"
								style="
									transform: translate(-50%, -50%);
									width: {gBrushToolSize - 2}px;
									height: {gBrushToolSize - 2}px;
									left: {gBrushOffsetX}px;
									top: {gBrushOffsetY}px;
									background-color: {gSelectedBrushMode === 'brush' ? '#408dff' : '#f5f5f5'};
									border: 1px solid {gSelectedBrushMode === 'brush' ? '#0261ed' : '#bfbfbf'};
									opacity: {gIsPainting ? 0.6 : 0.5};
									"
							/>
						</div>
						<canvas
							class="shadow-lg inset-0 w-full h-full
						{gAnythingEssentialLoading ? 'opacity-30 cursor-not-allowed' : ''} 
						{gShowOriginalImage === true ? '!hidden' : '!block'}
						
						"
							id="imageCanvas"
							bind:this={gImageCanvas}
						/>

						<canvas
							id="maskCanvas"
							class=" inset-0 w-full h-full absolute opacity-50
						{gPanEnabled ? '' : 'panzoom-exclude'} 
						{gAnythingEssentialLoading ? 'opacity-30 cursor-not-allowed' : ''}
						{gShowOriginalImage ? '!hidden' : '!block'}
						"
							bind:this={gMaskCanvas}
							on:mousedown={gSelectedTool === 'brush' && !gPanEnabled
								? (e) => startPainting(e, gMaskCanvas)
								: undefined}
							on:mouseup={gSelectedTool === 'brush' && !gPanEnabled ? stopPainting : undefined}
							on:mousemove={(event) =>
								gSelectedTool === 'brush' && !gPanEnabled
									? handleEditorCursorMove(event, gMaskCanvas)
									: undefined}
							on:touchstart={gSelectedTool === 'brush' && !gPanEnabled
								? (e) => startPainting(e, gMaskCanvas)
								: undefined}
							on:touchend={gSelectedTool === 'brush' && !gPanEnabled ? stopPainting : undefined}
							on:touchmove={gSelectedTool === 'brush' && !gPanEnabled
								? (e) => {
										e.preventDefault();
										handleEditorCursorMove(e, gMaskCanvas);
								  }
								: undefined}
							on:click={gSelectedTool === 'segment_anything' && !gPanEnabled
								? handleCanvasClick
								: undefined}
						/>

						<img
							class="shadow-lg inset-0 w-full h-full
								{gAnythingEssentialLoading ? 'opacity-50 cursor-not-allowed' : ''}
								{gShowOriginalImage === true ? '!block' : '!hidden'}"
							src={$uploadedImgBase64}
							alt="originalImage"
							bind:this={gOriginalImgElement}
						/>
					</div>
					<div class="flex-none" />
				</div>
			</div>
		</div>
		<!-- bottom buttons row -->
		<div class="flex flex-none flex-wrap lg:gap-x-2 gap-x-1">
			<div class="sm:flex-1 flex-0" />
			<div>
				<PanzoomCanvasControls
					disabled={gAnythingEssentialLoading}
					currentZoomValue={gCurrentZoom}
					bind:panEnabled={gPanEnabled}
					on:zoomIn={gPanzoomObj.zoomIn}
					on:zoomOut={gPanzoomObj.zoomOut}
					on:zoomReset={gPanzoomObj.reset}
				/>
			</div>
			<div class="flex-1 flex justify-end">
				<button
					class="btn lg:btn-xl md:btn-md btn-sm variant-filled-primary text-white dark:text-white font-semibold"
					disabled={gAnythingEssentialLoading || gInpainterLoading}
					on:click={handleInpainting}
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
