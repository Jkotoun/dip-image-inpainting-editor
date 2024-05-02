

type Point = { x: number; y: number };

//dilate SAM model output mask by a specified number of pixels for better results of inpainting
export async function dilateMaskByPixels(
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

//convert img data from mask canvas to binary mask
export function maskArrayFromImgData(imageData: ImageData, canvasWidth: number, canvasHeight: number): boolean[][] {
    const maskArray: boolean[][] = [];
    for (let y = 0; y < canvasHeight; y++) {
        const row: boolean[] = [];
        for (let x = 0; x < canvasWidth; x++) {
            const index = (y * canvasWidth + x) * 4; //RGBA
            const alpha = imageData.data[index + 3]; // Alpha value indicates if the pixel is drawn to
            row.push(alpha > 128); 
        }
        maskArray.push(row);
    }
    return maskArray;
}

//download canvas data as png
export function downloadImage(imageData: ImageData, imgName: string) {
    console.log("start download action")
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

export function clearCanvas(canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext('2d');
    if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
}