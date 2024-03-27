import { clearCanvas } from "$lib/editorHelpers";
import { booleanMaskToUint8Buffer, imgDataToRGBArray, reshapeBufferToNCHW, reshapeCHWtoHWC } from "$lib/onnxHelpers";
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
export async function renderEditorState(
    state: editorState,
    imageCanvas: HTMLCanvasElement,
    maskCanvas: HTMLCanvasElement,
    ImgResToCanvasSizeRatio: number
) {
    clearCanvas(imageCanvas);
    clearCanvas(maskCanvas);
    drawImage(imageCanvas, state.imgData);
    drawMarkers(imageCanvas,ImgResToCanvasSizeRatio, state.clickedPositions);
    drawMask(imageCanvas, state.maskSAMDilated, 0.5, false);
    drawMask(maskCanvas, state.maskBrush, 1, true);
}

function drawMarkers(canvas: HTMLCanvasElement, ImgResToCanvasSizeRatio: number,  clickedPositions: SAMmarker[]) {
    const canvasContext = canvas.getContext('2d');
    if (!canvasContext) return;
    for (const pos of clickedPositions) {
        canvasContext.fillStyle = pos.type === 'positive' ? '#021ded' : 'red';
        canvasContext.beginPath();
        canvasContext.arc(pos.x, pos.y, 5 * ImgResToCanvasSizeRatio, 0, Math.PI * 2);
        canvasContext.fill();
    }
}

export function drawMask(
    canvas: HTMLCanvasElement,
    maskArray: any,
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

export async function runInpainting(currentEditorState: editorState, worker: Worker): Promise<void> {
    let imgUInt8Array = imgDataToRGBArray(currentEditorState.imgData).rgbArray;
    let nchwBuffer = reshapeBufferToNCHW(
        imgUInt8Array,
        1,
        3,
        currentEditorState.imgData.width,
        currentEditorState.imgData.height
    );
    //combine currentstate brushmask and sammask with or
    let maskArrayCombined = currentEditorState.maskBrush.map((row: boolean[], y: number) =>
        row.map((val, x) => val || currentEditorState.maskSAMDilated[y][x])
    );

    let maskUInt8Buffer = booleanMaskToUint8Buffer(maskArrayCombined);
    worker.postMessage({
        type: MESSAGE_TYPES.INPAINTING_RUN,
        data: {
            imageTensorData: {
                data: nchwBuffer,
                dims: [1, 3, currentEditorState.imgData.height, currentEditorState.imgData.width]
            },

            maskTensorData: {
                data: maskUInt8Buffer,
                dims: [1, 1, currentEditorState.imgData.height, currentEditorState.imgData.width]
            }
        }
    });
}

export function createEmptyMaskArray(width: number, height: number) {
    return new Array(height).fill(false).map(() => new Array(width).fill(false));
}