export interface imgRGBData {
    rgbArray: Uint8Array;
    width: number;
    height: number;
}

    //EMBEDDING FUNCTIONS
	export async function getResizedImgData(
		img: ImageData,
		longSideLength: number = 1024
	): Promise<imgRGBData> {
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
		let tmpCanvasData = tempCanvas.getContext('2d')!.getImageData(0, 0, tempCanvas.width, tempCanvas.height)
        let rgbArray= imgDataToRGBArray(tmpCanvasData);
		return rgbArray;
	}

	export function imgDataToRGBArray(imgData: ImageData): imgRGBData {
		let pixels = imgData?.data;
		//create rgb array
		let rgbArray = new Uint8Array(imgData.width * imgData.height * 3);
		for (let i = 0; i < imgData.width * imgData.height; i++) {
			rgbArray[i * 3] = pixels![i * 4];
			rgbArray[i * 3 + 1] = pixels![i * 4 + 1];
			rgbArray[i * 3 + 2] = pixels![i * 4 + 2];
		}
		return { rgbArray, width: imgData.width, height: imgData.height } as imgRGBData;
	}

    export function reshapeBufferToNCHW(
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


    export function booleanMaskToUint8Buffer(maskArray: boolean[][]) {
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

    
	export function reshapeCHWtoHWC(
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