<script lang="ts">
	import { onMount } from 'svelte';
	import * as tf from '@tensorflow/tfjs';

	import Npyjs from 'npyjs';

	let imageURL = '';

    interface modelInputInterface {
        image_embeddings: tf.Tensor;
        point_coords: tf.Tensor;
        point_labels: tf.Tensor;
        mask_input: tf.Tensor;
        has_mask_input: tf.Tensor;
        orig_im_size: tf.Tensor;
    }

    let modelInput: any;
    let model:  tf.GraphModel<string | tf.io.IOHandler>;
	onMount(async () => {
		await tf.ready();
		try {
			await import('@tensorflow/tfjs-backend-webgpu');
			await tf.setBackend('webgpu');
			console.log(tf.env().getFlags());
		} catch (e) {
			try {
				await tf.setBackend('webgl');
			} catch (e) {
				console.error('could not load backend:', e);
				throw e;
			}
		}
		// Load the image from a static URL
		imageURL = './demobear.png';
		const response = await fetch(imageURL);
		const blob = await response.blob();
		const image = await createImageBitmap(blob);
  
		const npy = new Npyjs();
		const embeddings = await npy.load('image_embedding.npy');
		// console.log("embedingy")
		// console.log(embeddings.data[0])
		// @ts-ignore
		const embeddings_tensor = tf.tensor(embeddings.data, embeddings.shape, 'float32');

		// // Get the original size of the image
		const originalWidth = image.width;
		const originalHeight = image.height;

		// Your input point and label data
		const inputPoint = [
			[600, 300],
			[550, 450],
			[400, 450],
			[0, 0]
		];
		const inputLabels = [1, 1, 1, -1];

		// // Calculate scale factor
		const longSideLength = 1024;
		const scale = longSideLength / Math.max(originalWidth, originalHeight);

		let newHeight = Math.round(originalHeight * scale);
		let newWidth = Math.round(originalWidth * scale);

		const realScaleWidth = newWidth / originalWidth;
		const realScaleHeight = newHeight / originalHeight;
		const onnxInputPoints = inputPoint.map(([x, y]) => [x * realScaleWidth, y * realScaleHeight]);

		// // Create input tensors
		const pointCoordsTensor = tf.tensor(
			[onnxInputPoints],
			[1, onnxInputPoints.length, 2],
			'float32'
		);
		const pointLabelsTensor = tf.tensor([inputLabels], [1, inputLabels.length], 'float32');
		const origImgSizeTensor = tf.tensor([originalHeight, originalWidth], [2], 'float32');
		const maskInputTensor = tf.tensor(
			new Float32Array(256 * 256).fill(0),
			[1, 1, 256, 256],
			'float32'
		);
		const hasMaskInputTensor = tf.tensor([0], [1], 'float32');

        modelInput = {
            image_embeddings: embeddings_tensor,
            point_coords: pointCoordsTensor,
            point_labels: pointLabelsTensor,
            mask_input: maskInputTensor,
            has_mask_input: hasMaskInputTensor,
            orig_im_size: origImgSizeTensor
        };
        model = await tf.loadGraphModel('/tfjs_decoder_base/model.json');
	});

    async function runModel () {
        if (!modelInput) return;
		// let data = modelInput.image_embeddings.arraySync();
		const predictions: any = await model.executeAsync(modelInput);
        const lastData = await predictions[predictions.length - 1].arraySync();
		console.log(lastData[0][0][480][590])

		// let indices = []
		// const data = lastData[0][0];
		// for (let i = 0; i < data.length; i++) {
		// 	for (let j = 0; j < data[i].length; j++) {
		// 		if (data[i][j] > 0.0) {
		// 			indices.push([i, j]);
		// 		}
		// 	}
		// }

		// console.log(indices.length);

        // const arrayOfArrays = lastData[0][0];
		// console.log(arrayOfArrays[590][480]);

}
</script>

<!-- Your image can be displayed here if needed -->
<img src={imageURL} alt="Loaded" />
<button on:click={runModel}>Run Model</button>
<button on:click={() => console.log(modelInput)}>log input</button>
<!-- {#if modelInput}
    <p>Model input loaded</p>
    <button on:click={runModel}>Run Model</button>
{/if} -->

