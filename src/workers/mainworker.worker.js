import * as ort from 'onnxruntime-web';
// import * as tf from '@tensorflow/tfjs';
import { MESSAGE_TYPES } from './messageTypes';

const mobileSAMEncoderPath = '/mobile_sam.encoder.onnx';
const mobileSAMDecoderPath = '/tfjs_tiny_decoder_quantized/model.json';
const mobile_inpainting_GAN = '/migan_pipeline_v2.onnx';
const modelSAMDecoderONNXPath = '/sam_onnx_decoder_mobile_quantized.onnx';
let decoderReady = false;
let encoderReady = false;
let inpainterReady = false;
// @ts-ignore
let encoderOnnxSession;
// @ts-ignore
let miganOnnxSession;
// @ts-ignore
let tfjsDecoder;
// @ts-ignore
let decoderOnnxSession;


async function loadEncoderDecoder(){
    if(encoderReady && decoderReady){
        return;
    }
    encoderOnnxSession = await ort.InferenceSession.create(mobileSAMEncoderPath, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
    });
    decoderOnnxSession = await ort.InferenceSession.create(modelSAMDecoderONNXPath, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
    });
    encoderReady = true;
    // await tf.ready();
    // try {
    //     await import('@tensorflow/tfjs-backend-webgpu');
    //     await tf.setBackend('webgpu');
    // } catch (e) {
    //     try {
    //         await tf.setBackend('webgl');
    //     } catch (e) {
    //         console.error('could not load backend:', e);
    //         throw e;
    //     }
    // }
    // tfjsDecoder = await tf.loadGraphModel(mobileSAMDecoderPath);
    decoderReady = true;
    self.postMessage({ type: MESSAGE_TYPES.ENCODER_DECODER_LOADED, data: "Encoder and decoder loaded" });
}
async function loadInpainter(){
    if(!inpainterReady){
        miganOnnxSession = await ort.InferenceSession.create(mobile_inpainting_GAN, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });
        inpainterReady = true;
        self.postMessage({ type: MESSAGE_TYPES.INPAINTER_LOADED, data: "Inpainter loaded" });
    }
}

loadEncoderDecoder();



async function checkLoadState() {
    if (encoderReady && decoderReady) {
        if(inpainterReady){
            self.postMessage({ type: MESSAGE_TYPES.ALL_MODELS_LOADED, data: "All models loaded" });
        }
        else{
            self.postMessage({ type: MESSAGE_TYPES.ENCODER_DECODER_LOADED, data: "Encoder decoder loaded" });
        }
    }
    else {
        self.postMessage({ type: MESSAGE_TYPES.NONE_LOADED, data: "Models not loaded" });
    }
}

self.onmessage = async function (event) {
    const { type, data } = event.data;
    if (type === MESSAGE_TYPES.CHECK_MODELS_LOADING_STATE) {
        checkLoadState();
    }
    else if(type === MESSAGE_TYPES.LOAD_INPAINTER){
        if(inpainterReady)
            return;
        loadInpainter();
    }
    else if (type === MESSAGE_TYPES.DECODER_RUN && decoderReady) {
        const resultTest = await runDecoderONNX(data);
        self.postMessage({ type: MESSAGE_TYPES.DECODER_RUN_RESULT, data: resultTest });
    }
    else if (type === MESSAGE_TYPES.ENCODER_RUN && encoderReady) {
        const encoderResult = await runEncoder(data);
        self.postMessage({
            type: MESSAGE_TYPES.ENCODER_RUN_RESULT, data: {
                embeddings: encoderResult['image_embeddings'].data,
                dims: encoderResult['image_embeddings'].dims
            }
        });
    }
    else if (type === MESSAGE_TYPES.INPAINTING_RUN && inpainterReady) {
        const inpaintingResult = await runInpainting(data);
        self.postMessage({ type: MESSAGE_TYPES.INPAINTING_RUN_RESULT, data: inpaintingResult['result'].data });
    }
    else {
        self.postMessage({ type: MESSAGE_TYPES.TEST, data: "Response" });
    }
    
    // Respond back to the main thread
};


self.onerror = function (event) {
    console.error(event);
}

// @ts-ignore
async function runEncoder(data) {
    let inputTensor = new ort.Tensor('float32', data.img_array_data, data.dims)
    // @ts-ignore
    const output = await encoderOnnxSession.run({ input_image: inputTensor });
    return output
}

// // @ts-ignore
// async function runDecoder(data) {

//     let inputDict = {
//         "has_mask_input": tf.tensor(data.has_mask_input.data, data.has_mask_input.dims, 'float32'),
//         "image_embeddings": tf.tensor(data.image_embeddings.data, data.image_embeddings.dims, 'float32'),
//         "mask_input": tf.tensor(data.mask_input.data, data.mask_input.dims, 'float32'),
//         "orig_im_size": tf.tensor(data.orig_im_size.data, data.orig_im_size.dims, 'float32'),
//         "point_coords": tf.tensor(data.point_coords.data, data.point_coords.dims, 'float32'),
//         "point_labels": tf.tensor(data.point_labels.data, data.point_labels.dims, 'float32'),
//     }
//     // @ts-ignore
//     const predictions = await tfjsDecoder.executeAsync(inputDict);

//     // @ts-ignore
//     const lastData = await predictions[predictions.length - 1].arraySync();
//     return lastData[0][0];
// }

// @ts-ignore
async function runDecoderONNX(data){
    let inputDict = {
        has_mask_input: new ort.Tensor('float32', data.has_mask_input.data, data.has_mask_input.dims),
        image_embeddings: new ort.Tensor('float32', data.image_embeddings.data, data.image_embeddings.dims),
        mask_input: new ort.Tensor('float32', data.mask_input.data, data.mask_input.dims),
        orig_im_size: new ort.Tensor('float32', data.orig_im_size.data, data.orig_im_size.dims),
        point_coords: new ort.Tensor('float32', data.point_coords.data, data.point_coords.dims),
        point_labels: new ort.Tensor('float32', data.point_labels.data, data.point_labels.dims)
    }
    // @ts-ignore
    const output = await decoderOnnxSession?.run(inputDict);
    let mask = output['masks'].data
    let dims = output['masks'].dims
    const reshapedArray = reshapeFloat32ArrayTo2D(mask, dims[2], dims[3]);
    return reshapedArray
}
//@ts-ignore
function reshapeFloat32ArrayTo2D(array, numRows, numCols) {
    if (numRows * numCols !== array.length) {
        throw new Error('Number of elements in the array does not match the specified dimensions.');
    }

    const result = [];
    for (let i = 0; i < numRows; i++) {
        const row = [];
        for (let j = 0; j < numCols; j++) {
            row.push(array[i * numCols + j]);
        }
        result.push(row);
    }

    return result;
}


// @ts-ignore
async function runInpainting(data) {
    let maskTensor = new ort.Tensor('uint8', data.maskTensorData.data, data.maskTensorData.dims);
    let imageTensor = new ort.Tensor('uint8', data.imageTensorData.data, data.imageTensorData.dims);
    // @ts-ignore
    const output = await miganOnnxSession?.run({ image: imageTensor, mask: maskTensor });
    return output

}