import * as ort from 'onnxruntime-web';
import * as tf from '@tensorflow/tfjs';
import { MESSAGE_TYPES } from './messageTypes';

const mobileSAMEncoderPath = '/mobile_sam.encoder.onnx';
const mobileSAMDecoderPath = '/tfjs_tiny_decoder_quantized/model.json';
const mobile_inpainting_GAN = '/migan_pipeline_v2.onnx';
let decoderReady = false;
let encoderReady = false;
let inpainterReady = false;
// @ts-ignore
let encoderOnnxSession;
// @ts-ignore
let miganOnnxSession;
// @ts-ignore
let tfjsDecoder;


async function loadModels() {
    encoderOnnxSession = await ort.InferenceSession.create(mobileSAMEncoderPath, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
    });
    encoderReady = true;
    console.log("encoder loaded")
    await tf.ready();
    try {
        await import('@tensorflow/tfjs-backend-webgpu');
        await tf.setBackend('webgpu');
        console.log('webgpu loaded');
    } catch (e) {
        try {
            await tf.setBackend('webgl');
        } catch (e) {
            console.error('could not load backend:', e);
            throw e;
        }
    }
    tfjsDecoder = await tf.loadGraphModel(mobileSAMDecoderPath);
    decoderReady = true;
    console.log("decoder loaded")


    miganOnnxSession = await ort.InferenceSession.create(mobile_inpainting_GAN, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
    });
    inpainterReady = true;
    console.log("inpainter loaded")
}

loadModels();    



self.onmessage = async function (event) {
    const { type, data } = event.data;
    // console.log(`recieved  ${type} type of message`);

    if (type === MESSAGE_TYPES.DECODER_RUN) {
        const decoderResult = await runDecoder(data);
        self.postMessage({ type: MESSAGE_TYPES.DECODER_RUN_RESULT, data: decoderResult });
    }
    else if (type === MESSAGE_TYPES.ENCODER_RUN) {
        const encoderResult = await runEncoder(data);
        self.postMessage({
            type: MESSAGE_TYPES.ENCODER_RUN_RESULT, data: {
                embeddings: encoderResult['image_embeddings'].data,
                dims: encoderResult['image_embeddings'].dims
            }
        });
    }
    else if (type === MESSAGE_TYPES.INPAINTING_RUN) {
        const inpaintingResult = await runInpainting(data);
        self.postMessage({ type: MESSAGE_TYPES.INPAINTING_RUN_RESULT, data: inpaintingResult['result'].data });
    }
    else {
        console.log(data)
        self.postMessage({ type: MESSAGE_TYPES.TEST, data: "Response" });
    }
    // Respond back to the main thread
};

// @ts-ignore
async function runEncoder(data) {
    let inputTensor = new ort.Tensor('float32', data.img_array_data, data.dims)
    // @ts-ignore
    const output = await encoderOnnxSession.run({ input_image: inputTensor });
    return output
}

// @ts-ignore
async function runDecoder(data) {

    let inputDict = {
        "has_mask_input": tf.tensor(data.has_mask_input.data, data.has_mask_input.dims, 'float32'),
        "image_embeddings": tf.tensor(data.image_embeddings.data, data.image_embeddings.dims, 'float32'),
        "mask_input": tf.tensor(data.mask_input.data, data.mask_input.dims, 'float32'),
        "orig_im_size": tf.tensor(data.orig_im_size.data, data.orig_im_size.dims, 'float32'),
        "point_coords": tf.tensor(data.point_coords.data, data.point_coords.dims, 'float32'),
        "point_labels": tf.tensor(data.point_labels.data, data.point_labels.dims, 'float32'),
    }
    // @ts-ignore
    const predictions = await tfjsDecoder.executeAsync(inputDict);

    // @ts-ignore
    const lastData = await predictions[predictions.length - 1].arraySync();
    return lastData[0][0];
}


// @ts-ignore
async function runInpainting(data) {
    let maskTensor = new ort.Tensor('uint8', data.maskTensorData.data, data.maskTensorData.dims);
    let imageTensor = new ort.Tensor('uint8', data.imageTensorData.data, data.imageTensorData.dims);
    // @ts-ignore
    const output = await miganOnnxSession?.run({ image: imageTensor, mask: maskTensor });
    return output

}