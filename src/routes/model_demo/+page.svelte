<script>
// @ts-nocheck

    import { onMount } from 'svelte';
    import * as tf from '@tensorflow/tfjs';
    import '@tensorflow/tfjs-backend-webgl'; // Load WebGL backend
    import '@tensorflow/tfjs-converter'; // Load model converter
    import { IMAGENET_CLASSES } from './imagenet_classes';
    let model;
    let imageFile;
    let predictedClass = 0;


    let isLoading = true;
    const MOBILENET_MODEL_PATH =
    'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_100_224/classification/3/default/1';

    const IMAGE_SIZE = 224;
    

    async function loadModel() {
        try {
        model = await tf.loadGraphModel(MOBILENET_MODEL_PATH, { fromTFHub: true })

        console.log("model_loaded")
        isLoading = false; // Model loaded successfully
      } catch (error) {
        console.error('Error loading the model:', error);
        isLoading = false; // Set loading state to false even on error
      }
    }
  
    async function classifyImage() {
    console.log('Image loaded');

    const img = new Image();
    img.src = URL.createObjectURL(imageFile);
    await img.decode();

    const imgTensor = tf.tidy(() => {
        // tf.browser.fromPixels() returns a Tensor from an image element.
        const tempImgTensor = tf.browser.fromPixels(img).toFloat();

        // Normalize the image from [0, 255] to [-1, 1].
        const offset = tf.scalar(127.5);
        const normalized = tempImgTensor.sub(offset).div(offset);

        return normalized;
    });

    // Resize the image if its dimensions are not equal to IMAGE_SIZE.
    const resizedImgTensor = tf.image.resizeBilinear(imgTensor, [IMAGE_SIZE, IMAGE_SIZE]);

    const batched = resizedImgTensor.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);

    // Make a prediction through the model.
    const pred = model.predict(batched);

    // Get top class.
    const topClass = pred.argMax(1).dataSync()[0];

    // Clean up tensors.
    imgTensor.dispose();
    resizedImgTensor.dispose();

    predictedClass = topClass;
}
  
    onMount(loadModel); // Load the model when the component mounts
  </script>
  
  {#if isLoading}
    <p>Loading model...</p>
  {:else}
  <h1>Image Classification App</h1>
  
  <input type="file" accept="image/*" on:change={(e) => (imageFile = e.target.files[0])} />
  <button on:click={classifyImage}>Classify</button>
  {#if predictedClass !== ''}
  <p>{IMAGENET_CLASSES[predictedClass]}</p>
  
  {/if}
  {/if}
  <style>
    /* Add your styles here */
  </style>
  