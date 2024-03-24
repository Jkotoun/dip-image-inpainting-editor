//@ts-nocheck
let ort;

let mobileSAMEncoderPath = "/mobile_sam.encoder.onnx";
let modelSAMDecoderONNXPath = "/sam_onnx_decoder_mobile_quantized.onnx";
let mobile_inpainting_GAN = "/migan_pipeline_v2.onnx";
function init(env) {
    console.log(env)
    if(env !== "development"){
        mobileSAMEncoderPath = appBasePath + mobileSAMEncoderPath;
        modelSAMDecoderONNXPath = appBasePath + modelSAMDecoderONNXPath;
        mobile_inpainting_GAN = appBasePath + mobile_inpainting_GAN;
    }
    import("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/esm/ort.webgpu.min.js")
        .then(module => {
            ort = module.default;
            ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
            ort.env.logLevel = 'fatal';
            loadEncoderDecoder();
        })
        .catch(error => {
            console.error("Error loading onnxruntime-web:", error);
        });
}


import { MESSAGE_TYPES } from './messageTypes';
// let dev = false;
// const mobileSAMEncoderPath = `${dev ? "" : "/dip-image-inpainting-editor"}/mobile_sam.encoder.onnx`;
// //good results, but doesnt work with webgpu (wrong results), much slower on wasm (approx 3x)
// // const mobileSAMEncoderPath = '/mobile_sam.encoder_fp16.onnx';

// const mobile_inpainting_GAN = `${dev ? "" : "/dip-image-inpainting-editor"}/migan_pipeline_v2.onnx`;
// //doesnt work with webgpu (throws error), much slower on wasm (approx 3x), also slightly worse results
// // const mobile_inpainting_GAN = '/migan_pipeline_v2_fp16.onnx';


// const modelSAMDecoderONNXPath = `${dev ? "" : "/dip-image-inpainting-editor"}/sam_onnx_decoder_mobile_quantized.onnx`;
let decoderReady = false;
let encoderReady = false;
let inpainterReady = false;

let encoderOnnxSession;

let miganOnnxSession;

let decoderOnnxSession;

async function loadEncoderDecoder() {
    if (encoderReady && decoderReady) {
        return;
    }
    try {
        encoderOnnxSession = await ort.InferenceSession.create(mobileSAMEncoderPath, {
            executionProviders: ['webgpu'],
            graphOptimizationLevel: 'all',
        });
        console.log("init SAM encoder with webgpu succeeded")
    }
    catch (e) {
        console.log("couldnt init SAM encoder with webgpu, trying wasm")
        encoderOnnxSession = await ort.InferenceSession.create(mobileSAMEncoderPath, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });
    }
    decoderOnnxSession = await ort.InferenceSession.create(modelSAMDecoderONNXPath, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
    });
    encoderReady = true;
    decoderReady = true;
    self.postMessage({ type: MESSAGE_TYPES.ENCODER_DECODER_LOADED, data: "Encoder and decoder loaded" });
}
async function loadInpainter() {
    if (!inpainterReady) {
        try {
            miganOnnxSession = await ort.InferenceSession.create(mobile_inpainting_GAN, {
                executionProviders: ['webgpu'],
                graphOptimizationLevel: 'disabled',
            });
            console.log("webgpu migan init succeeded")
        }
        catch (e) {
            console.log("webgpu migan init failed, trying wasm")
            miganOnnxSession = await ort.InferenceSession.create(mobile_inpainting_GAN, {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all'
            });
        }
        inpainterReady = true;
        self.postMessage({ type: MESSAGE_TYPES.INPAINTER_LOADED, data: "Inpainter loaded" });
    }
}


async function checkLoadState() {
    if (encoderReady && decoderReady) {
        if (inpainterReady) {
            self.postMessage({ type: MESSAGE_TYPES.ALL_MODELS_LOADED, data: "All models loaded" });
        }
        else {
            self.postMessage({ type: MESSAGE_TYPES.ENCODER_DECODER_LOADED, data: "Encoder decoder loaded" });
        }
    }
    else {
        self.postMessage({ type: MESSAGE_TYPES.NONE_LOADED, data: "Models not loaded" });
    }
}

self.onmessage = async function (event) {
    const { type, data } = event.data;
    if(type === MESSAGE_TYPES.INIT){
        init(data.env)
    }
    else if (type === MESSAGE_TYPES.CHECK_MODELS_LOADING_STATE) {
        checkLoadState();
    }
    else if (type === MESSAGE_TYPES.LOAD_INPAINTER) {
        if (inpainterReady)
            return;
        loadInpainter();
    }
    else if (type === MESSAGE_TYPES.DECODER_RUN && decoderReady) {
        let startTime = performance.now();
        const resultTest = await runDecoderONNX(data);
        console.log('Time taken for decoder run:', performance.now() - startTime, 'ms');
        self.postMessage({ type: MESSAGE_TYPES.DECODER_RUN_RESULT, data: resultTest });
    }
    else if (type === MESSAGE_TYPES.ENCODER_RUN && encoderReady) {
        let startTime = performance.now();
        const encoderResult = await runEncoder(data);
        console.log('Time taken for encoder run:', performance.now() - startTime, 'ms');
        self.postMessage({
            type: MESSAGE_TYPES.ENCODER_RUN_RESULT, data: {
                embeddings: encoderResult['image_embeddings'].data,
                dims: encoderResult['image_embeddings'].dims
            }
        });
    }
    else if (type === MESSAGE_TYPES.INPAINTING_RUN && inpainterReady) {
        let startTime = performance.now();
        const inpaintingResult = await runInpainting(data);
        console.log('Time taken for inpainting run:', performance.now() - startTime, 'ms');
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


async function runEncoder(data) {
    let inputTensor = new ort.Tensor('float32', data.img_array_data, data.dims)

    const output = await encoderOnnxSession.run({ input_image: inputTensor });
    return output
}

// 
// async function runDecoder(data) {

//     let inputDict = {
//         "has_mask_input": tf.tensor(data.has_mask_input.data, data.has_mask_input.dims, 'float32'),
//         "image_embeddings": tf.tensor(data.image_embeddings.data, data.image_embeddings.dims, 'float32'),
//         "mask_input": tf.tensor(data.mask_input.data, data.mask_input.dims, 'float32'),
//         "orig_im_size": tf.tensor(data.orig_im_size.data, data.orig_im_size.dims, 'float32'),
//         "point_coords": tf.tensor(data.point_coords.data, data.point_coords.dims, 'float32'),
//         "point_labels": tf.tensor(data.point_labels.data, data.point_labels.dims, 'float32'),
//     }
//     
//     const predictions = await tfjsDecoder.executeAsync(inputDict);

//     
//     const lastData = await predictions[predictions.length - 1].arraySync();
//     return lastData[0][0];
// }


async function runDecoderONNX(data) {
    let inputDict = {
        has_mask_input: new ort.Tensor('float32', data.has_mask_input.data, data.has_mask_input.dims),
        image_embeddings: new ort.Tensor('float32', data.image_embeddings.data, data.image_embeddings.dims),
        mask_input: new ort.Tensor('float32', data.mask_input.data, data.mask_input.dims),
        orig_im_size: new ort.Tensor('float32', data.orig_im_size.data, data.orig_im_size.dims),
        point_coords: new ort.Tensor('float32', data.point_coords.data, data.point_coords.dims),
        point_labels: new ort.Tensor('float32', data.point_labels.data, data.point_labels.dims)
    }

    const output = await decoderOnnxSession?.run(inputDict);
    let mask = output['masks'].data
    let dims = output['masks'].dims
    const reshapedArray = reshapeFloat32ArrayTo2D(mask, dims[2], dims[3]);
    return reshapedArray
}

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



async function runInpainting(data) {
    let maskTensor = new ort.Tensor('uint8', data.maskTensorData.data, data.maskTensorData.dims);
    let imageTensor = new ort.Tensor('uint8', data.imageTensorData.data, data.imageTensorData.dims);

    const output = await miganOnnxSession?.run({ image: imageTensor, mask: maskTensor });
    return output

}