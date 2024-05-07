import { clearCanvas } from "$lib/editorHelpers";
import { booleanMaskToUint8Buffer, imgDataToRGBArray, reshapeBufferToNCHW, reshapeCHWtoHWC, type imgRGBData } from "$lib/onnxHelpers";
import type { tool } from "../../types/editorTypes";
import { MESSAGE_TYPES } from "../../workers/messageTypes";

export interface SAMmarker {
    x: number;
    y: number;
    type: 'positive' | 'negative';
}

export interface editorState {
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

export function drawImage(canvas: HTMLCanvasElement, imageData: ImageData) {
    canvas.getContext('2d')!.putImageData(imageData, 0, 0);
}

/**
 * renders editor state - draw image, mask from both brush and SAM and markers on clicked positions
 * @param state state to render
 * @param imageCanvas reference to canvas for image rendering
 * @param maskCanvas reference to canvas for mask rendering
 * @param ImgResToCanvasSizeRatio ratio of image resolution to canvas element size
 * @returns editor state rendering promise
 */
export async function renderEditorState(
    state: editorState,
    imageCanvas: HTMLCanvasElement,
    maskCanvas: HTMLCanvasElement,
    ImgResToCanvasSizeRatio: number
) {
    return new Promise<void>((resolve) => {
        clearCanvas(imageCanvas);
        clearCanvas(maskCanvas);
        drawImage(imageCanvas, state.imgData);
        drawMarkers(imageCanvas, ImgResToCanvasSizeRatio, state.clickedPositions);
        drawMask(imageCanvas, state.maskSAMDilated, 0.5, false);
        drawMask(maskCanvas, state.maskBrush, 1, true);
        resolve();
    })
}
/**
 * draw smart selector tool markers on clicked positions 
 * @param canvas canvas to draw marks to
 * @param ImgResToCanvasSizeRatio ratio of image resolution to canvas element size
 * @param clickedPositions positions in image coordinates to draw to canvas
 */
function drawMarkers(canvas: HTMLCanvasElement, ImgResToCanvasSizeRatio: number, clickedPositions: SAMmarker[]) {
    const canvasContext = canvas.getContext('2d');
    if (!canvasContext) return;
    for (const pos of clickedPositions) {
        canvasContext.fillStyle = pos.type === 'positive' ? '#021ded' : 'red';
        canvasContext.beginPath();
        canvasContext.arc(pos.x, pos.y, 5 * ImgResToCanvasSizeRatio, 0, Math.PI * 2);
        canvasContext.fill();
    }
}

/**
 * draw mask on canvas from 2D binary mask array
 * @param canvas canvas to draw mask to
 * @param maskArray mask array to draw
 * @param opacity opacity of mask in canvas
 * @param clearCanvasFirst whether to clear canvas before drawing mask
 */
export function drawMask(
    canvas: HTMLCanvasElement,
    maskArray: boolean[][],
    opacity: number,
    clearCanvasFirst = false
) {
    const canvasCtx = canvas.getContext('2d');
    if (canvasCtx) {
        if (clearCanvasFirst) {
            clearCanvas(canvas);
        }
        const prevMode = canvasCtx.globalCompositeOperation;
        canvasCtx.globalCompositeOperation = 'source-over';
        canvasCtx.fillStyle = `rgba(64, 141, 255, ${opacity})`;

        for (let y = 0; y < maskArray.length; y++) {
            for (let x = 0; x < maskArray[y].length; x++) {
                if (maskArray[y][x]) {
                    canvasCtx.fillRect(x, y, 1, 1);
                }
            }
        }
        canvasCtx.globalCompositeOperation = prevMode;
    }
}

/**
 * MI-GAN inpainter postprocessing - convert CHW output to canvas imageData to display result
 * @param imageDataRGB source rgb (chw) uint8 array
 * @param img_height source image height
 * @param img_width source image width
 * @returns ImageData object to display on canvas
 */
export async function RGB_CHW_array_to_imageData(
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

/**
 * draw path from last to current position on canvas
 * @param x current X position
 * @param y current Y position
 * @param prevX previous X position
 * @param prevY previous Y position
 * @param brushSize brush size in pixels 
 * @param canvasContext canvas 2d context 
 * @param ImgResToCanvasSizeRatio ratio of image resolution to canvas element size used to scale the brush size 
 */
export function canvasBrushDraw(
    x: number,
    y: number,
    prevX: number,
    prevY: number,
    brushSize: number,
    canvasContext: CanvasRenderingContext2D,
    ImgResToCanvasSizeRatio: number
): { currentX: number; currentY: number } {
    //scale to website pixels from canvas res
    let brushSizeScaled = brushSize * ImgResToCanvasSizeRatio;
    // Draw on canvas
    if (prevX === x && prevY === y) {
        canvasContext.beginPath();
        canvasContext.arc(x, y, brushSizeScaled / 2, 0, 2 * Math.PI);
        canvasContext.fillStyle = 'rgba(89, 156, 255, 1)';
        canvasContext.fill();
    } else {
        canvasContext.strokeStyle = 'rgba(89, 156, 255, 1)';
        canvasContext.beginPath();
        canvasContext.lineJoin = 'round';
        canvasContext.lineCap = 'round';
        canvasContext.lineWidth = brushSizeScaled;
        canvasContext.moveTo(prevX, prevY);
        canvasContext.lineTo(x, y);
        //color
        canvasContext.closePath();
        canvasContext.stroke();
    }
    return { currentX: x, currentY: y };
}

/**
 * create mask and image buffer from current editor state and send message to webworker
 * @param currentEditorState current editor state
 * @param worker webworker object to send message to
 */
export async function runInpainting(currentEditorState: editorState, worker: Worker): Promise<void> {
    let imageNCHWBuffer = reshapeBufferToNCHW(
        imgDataToRGBArray(currentEditorState.imgData).rgbArray,
        1,
        3,
        currentEditorState.imgData.width,
        currentEditorState.imgData.height
    );

    //combine currentstate brushmask and sammask with 'or' operation
    let maskArrayCombined = currentEditorState.maskBrush.map((row: boolean[], y: number) =>
        row.map((val, x) => val || currentEditorState.maskSAMDilated[y][x])
    );
    let maskUInt8Buffer = booleanMaskToUint8Buffer(maskArrayCombined);
    worker.postMessage({
        type: MESSAGE_TYPES.INPAINTING_RUN,
        data: {
            imageTensorData: {
                data: imageNCHWBuffer,
                dims: [1, 3, currentEditorState.imgData.height, currentEditorState.imgData.width]
            },

            maskTensorData: {
                data: maskUInt8Buffer,
                dims: [1, 1, currentEditorState.imgData.height, currentEditorState.imgData.width]
            }
        }
    });
}

/**
 * create empty mask array filled with false value with given width and height
 * @param width width of mask
 * @param height height of mask
 * @returns empty mask array
 */
export function createEmptyMaskArray(width: number, height: number) {
    return new Array(height).fill(false).map(() => new Array(width).fill(false));
}

/**
 * create decoder input from current editor state (uploaded img + clicked positive/negative points)
 * @param currentEditorState current state of editor
 * @param resizedImgWidth width of image after resizing to long length 1024
 * @param resizedImgHeight height of image after resizing to long length 1024
 * @returns input dictionary for decoder model
 */
export async function createDecoderInputDict(currentEditorState: editorState, resizedImgWidth: number, resizedImgHeight: number) {

    function coordsToResizedImgScale(x: number, y: number) {
        const imageX = (x / currentEditorState.imgData.width) * resizedImgWidth;
        const imageY = (y / currentEditorState.imgData.height) * resizedImgHeight;
        return { x: imageX, y: imageY };
    }

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
            data: [currentEditorState.imgData.height, currentEditorState.imgData.width],
            dims: [2]
        }
    };
    return modelInput;
}


/**
 * convert rgb array to float buffer and send message to webworker
 * @param resizedImgData rgb array buffer of resized image data
 * @param worker webworker object to send message to
 */
export async function runModelEncoder(resizedImgData: imgRGBData, worker: Worker): Promise<void> {
    let floatArray = Float32Array.from(resizedImgData.rgbArray);
    worker.postMessage({
        type: MESSAGE_TYPES.ENCODER_RUN,
        data: {
            img_array_data: floatArray,
            dims: [resizedImgData.height, resizedImgData.width, 3]
        }
    });
}

/**
 * get current cursor when hovered over canvas based on current state
 * @param enablePan whether pan is enabled
 * @param selectedTool currently selected tool
 * @returns cursor to display on canvas
 */
export const currentCanvasCursor = (enablePan: boolean, selectedTool: tool) => {
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

/**
 * mask decoder result to 2D binary mask array - values >0 are considered as object pixels
 * @param decoderResult result of decoder model
 * @returns 2D binary mask array determining object pixels
 */
export const decoderResultToMaskArray = (decoderResult: number[][]) : boolean[][]=> {
    return decoderResult.map((val: number[]) => val.map((v) => v > 0.0));
};