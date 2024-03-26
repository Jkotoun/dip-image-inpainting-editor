//@ts-nocheck
let ort;
//used from https://stackoverflow.com/a/11381730
const checkMobile = () => {
    let check = false;
    (function (a) { if (/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino|android|ipad|playbook|silk/i.test(a) || /1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(a.substr(0, 4))) check = true; })(navigator.userAgent || navigator.vendor || window.opera);
    return check;
};

//temporary - webGPU is experimental feature, currently it is supported on some mobile browsers,
//but it is not stable, it unexpectedly crashes on some devices. So for now, wasm is used if mobile device is detected
let isMobile = checkMobile();
let mobileSAMEncoderPath = "/mobile_sam.encoder.onnx";
let modelSAMDecoderONNXPath = "/sam_onnx_decoder_mobile_quantized.onnx";
let mobile_inpainting_GAN = "/migan_pipeline_v2.onnx";
function init(env, appBasePath) {
    console.log(env)
    console.log("is mobile? : ", isMobile)
    if (env !== "development") {
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

let encoderOnnxCurrentProvider;
let inpainterOnnxCurrentProvider;

async function loadEncoderDecoder() {
    if (encoderReady && decoderReady) {
        return;
    }
    try {
        encoderOnnxSession = await ort.InferenceSession.create(mobileSAMEncoderPath, {
            executionProviders: isMobile ? ['wasm'] : ['webgpu'],
            graphOptimizationLevel: 'all',
        });
        encoderOnnxCurrentProvider = 'webgpu';
        console.log("init SAM encoder with webgpu succeeded")
    }
    catch (e) {
        console.log("couldnt init SAM encoder with webgpu, trying wasm")
        encoderOnnxSession = await ort.InferenceSession.create(mobileSAMEncoderPath, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });
        encoderOnnxCurrentProvider = 'wasm';
    }
    decoderOnnxSession = await ort.InferenceSession.create(modelSAMDecoderONNXPath, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
    });
    encoderReady = true;
    decoderReady = true;
    console.log(encoderOnnxSession)
    console.log(decoderOnnxSession)
    self.postMessage({ type: MESSAGE_TYPES.ENCODER_DECODER_LOADED, data: "Encoder and decoder loaded" });
}
async function loadInpainter() {
    if (!inpainterReady) {
        try {
            miganOnnxSession = await ort.InferenceSession.create(mobile_inpainting_GAN, {
                executionProviders: isMobile? ['wasm'] : ['webgpu'],
                graphOptimizationLevel: 'disabled',
            });
            inpainterOnnxCurrentProvider = 'webgpu';
            console.log("webgpu migan init succeeded")
        }
        catch (e) {
            console.log("webgpu migan init failed, trying wasm")
            miganOnnxSession = await ort.InferenceSession.create(mobile_inpainting_GAN, {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all'
            });
            inpainterOnnxCurrentProvider = 'wasm';
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
    if (type === MESSAGE_TYPES.INIT) {
        init(data.env, data.appBasePath)
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
        let decoderResult;
        try {

            decoderResult = await runDecoderONNX(data);
            console.log('Time taken for decoder run:', performance.now() - startTime, 'ms');
            self.postMessage({ type: MESSAGE_TYPES.DECODER_RUN_RESULT_SUCCESS, data: decoderResult });
        }
        catch (e) {
            console.log("crashed decoder run")
            console.error(e)
            self.postMessage({ type: MESSAGE_TYPES.DECODER_RUN_RESULT_FAILURE, data: "Error running decoder" });
        }
    }
    else if (type === MESSAGE_TYPES.ENCODER_RUN && encoderReady) {
        let startTime = performance.now();
        let encoderResult;
        try {
            encoderResult = await runEncoder(data);
            self.postMessage({
                type: MESSAGE_TYPES.ENCODER_RUN_RESULT_SUCCESS, data: {
                    embeddings: encoderResult['image_embeddings'].data,
                    dims: encoderResult['image_embeddings'].dims
                }
            });
            encoderResult['image_embeddings'].dispose()
            console.log('Time taken for encoder run:', performance.now() - startTime, 'ms');
        }
        catch (e) {
            console.log("crashed encoder run")
            console.error(e)
            self.postMessage({ type: MESSAGE_TYPES.ENCODER_RUN_RESULT_FAILURE, data: "Error running encoder" });
        }

    }
    else if (type === MESSAGE_TYPES.INPAINTING_RUN && inpainterReady) {
        let startTime = performance.now();
        let inpaintingResult;
        try {
            inpaintingResult = await runInpainting(data);
            console.log('Time taken for inpainting run:', performance.now() - startTime, 'ms');
            self.postMessage({ type: MESSAGE_TYPES.INPAINTING_RUN_RESULT_SUCCESS, data: inpaintingResult['result'].data });
            inpaintingResult['result'].dispose();
        }
        catch (e) {
            console.log("crashed inpainting run")
            console.error(e)
            self.postMessage({ type: MESSAGE_TYPES.INPAINTING_RUN_RESULT_FAILURE, data: "Error running inpainter" });
        }
    }

    // Respond back to the main thread
};


self.onerror = function (event) {
    console.error(event);
}


async function runEncoder(data) {
    let inputTensor = new ort.Tensor('float32', data.img_array_data, data.dims)

    const output = await encoderOnnxSession.run({ input_image: inputTensor });
    inputTensor.dispose();
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
    //dispose input tensors
    for (const key in inputDict) {
        inputDict[key].dispose();
    }
    let mask = output['masks'].data
    let dims = output['masks'].dims
    const reshapedArray = reshapeFloat32ArrayTo2D(mask, dims[2], dims[3]);
    output['masks'].dispose();
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
    maskTensor.dispose();
    imageTensor.dispose();
    return output

}

