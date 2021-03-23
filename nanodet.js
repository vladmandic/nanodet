const fs = require('fs');
const path = require('path');
const process = require('process');
const log = require('@vladmandic/pilogger');
const tf = require('@tensorflow/tfjs-node');
const canvas = require('canvas');
const { labels } = require('./coco-labels');

const modelOptions = {
  modelPath: 'file://model-tfjs-graph-m/nanodet.json',
  minScore: 0.10, // low confidence, but still remove irrelevant
  iouThreshold: 0.10, // be very aggressive with removing overlapped boxes
  maxResults: 20, // high number of results, but likely never reached
  scaleBox: 2.5, // increase box size
};

// save image with processed results
async function saveImage(img, res) {
  // create canvas
  const c = new canvas.Canvas(img.inputShape[0], img.inputShape[1]);
  const ctx = c.getContext('2d');
  ctx.lineWidth = 2;
  ctx.strokeStyle = 'white';
  ctx.fillStyle = 'white';
  ctx.font = 'small-caps 20px "Segoe UI"';

  // load and draw original image
  const original = await canvas.loadImage(img.fileName);
  ctx.drawImage(original, 0, 0, c.width, c.height);

  // draw all detected objects
  for (const obj of res) {
    // draw label at center
    ctx.fillText(`${Math.round(100 * obj.score)}% [${obj.strideSize}] #${obj.class} ${obj.label}`, obj.box[0] + 4, obj.box[1] + 20);
    // draw rect using x,y,h,w
    ctx.rect(obj.box[0], obj.box[1], obj.box[2] - obj.box[0], obj.box[3] - obj.box[1]);
  }
  ctx.stroke();

  // write canvas to jpeg
  const outImage = `outputs/${path.basename(img.fileName)}`;
  const out = fs.createWriteStream(outImage);
  out.on('finish', () => log.state('Created output image:', outImage));
  out.on('error', (err) => log.error('Error creating image:', outImage, err));
  const stream = c.createJPEGStream({ quality: 0.6, progressive: true, chromaSubsampling: true });
  stream.pipe(out);
}

// load image from file and prepares image tensor that fits the model
async function loadImage(fileName, inputSize) {
  const data = fs.readFileSync(fileName);
  const obj = tf.tidy(() => {
    const buffer = tf.node.decodeImage(data);
    const resize = tf.image.resizeBilinear(buffer, [inputSize, inputSize]);
    const cast = resize.cast('float32');
    const normalize = cast.div(255);
    const expand = normalize.expandDims(0);
    const transpose = expand.transpose([0, 3, 1, 2]);
    const tensor = transpose;
    const img = { fileName, tensor, inputShape: [buffer.shape[1], buffer.shape[0]], outputShape: tensor.shape, size: buffer.size };
    return img;
  });
  return obj;
}

// process model results
async function processResults(res, inputSize, outputShape) {
  let results = [];
  for (const strideSize of [1, 2, 4]) { // try each stride size as it detects large/medium/small objects
    // find scores, boxes, classes
    tf.tidy(() => { // wrap in tidy to automatically deallocate temp tensors
      const baseSize = strideSize * 13; // 13x13=169, 26x26=676, 52x52=2704
      // find boxes and scores output depending on stride
      // log.info('Variation:', strideSize, 'strides', baseSize, 'baseSize');
      const scores = res.find((a) => (a.shape[1] === (baseSize ** 2) && a.shape[2] === 80))?.squeeze();
      const features = res.find((a) => (a.shape[1] === (baseSize ** 2) && a.shape[2] === 32))?.squeeze();
      // log.state('Found features tensor:', features?.shape);
      // log.state('Found scores tensor:', scores?.shape);
      const scoreIdx = scores.argMax(1).dataSync(); // location of highest scores
      const scoresMax = scores.max(1).dataSync(); // values of highest scores
      const boxesMax = features.reshape([-1, 4, 8]); // reshape [32] to [4,8] where 8 is change of different features inside stride
      const boxIdx = boxesMax.argMax(2).arraySync(); // what we need is indexes of features with highest scores, not values itself
      for (let i = 0; i < scores.shape[0]; i++) {
        if (scoreIdx[i] !== 0 && scoresMax[i] > modelOptions.minScore) {
          const cx = (0.5 + Math.trunc(i % baseSize)) / baseSize; // center.x normalized to range 0..1
          const cy = (0.5 + Math.trunc(i / baseSize)) / baseSize; // center.y normalized to range 0..1
          const boxOffset = boxIdx[i].map((a) => a * (baseSize / strideSize / inputSize)); // just grab indexes of features with highest scores
          let boxRaw = [ // results normalized to range 0..1
            cx - (modelOptions.scaleBox / strideSize * boxOffset[0]),
            cy - (modelOptions.scaleBox / strideSize * boxOffset[1]),
            cx + (modelOptions.scaleBox / strideSize * boxOffset[2]),
            cy + (modelOptions.scaleBox / strideSize * boxOffset[3]),
          ];
          boxRaw = boxRaw.map((a) => Math.max(0, Math.min(a, 1))); // fix out-of-bounds coords
          const box = [ // results normalized to input image pixels
            boxRaw[0] * outputShape[0],
            boxRaw[1] * outputShape[1],
            boxRaw[2] * outputShape[0],
            boxRaw[3] * outputShape[1],
          ];
          const result = {
            score: scoresMax[i],
            strideSize,
            class: scoreIdx[i] + 1,
            label: labels[scoreIdx[i]].label,
            center: [Math.trunc(outputShape[0] * cx), Math.trunc(outputShape[1] * cy)],
            centerRaw: [cx, cy],
            box: box.map((a) => Math.trunc(a)),
            boxRaw,
          };
          results.push(result);
        }
      }
    });
  }

  // deallocate tensors
  res.forEach((t) => tf.dispose(t));

  // normally nms is run on raw results, but since boxes need to be calculated this way we skip calulcation of
  // unnecessary boxes and run nms only on good candidates (basically it just does IOU analysis as scores are already filtered)
  const nmsBoxes = results.map((a) => a.boxRaw);
  const nmsScores = results.map((a) => a.score);
  const nms = await tf.image.nonMaxSuppressionAsync(nmsBoxes, nmsScores, modelOptions.maxResults, modelOptions.iouThreshold, modelOptions.minScore);
  const nmsIdx = nms.dataSync();
  tf.dispose(nms);

  // filter & sort results
  results = results
    .filter((a, idx) => nmsIdx.includes(idx))
    .sort((a, b) => b.score - a.score);

  return results;
}

async function main() {
  log.header();

  // init tensorflow
  await tf.enableProdMode();
  await tf.setBackend('tensorflow');
  await tf.ENV.set('DEBUG', false);
  await tf.ready();

  // load model
  const model = await tf.loadGraphModel(modelOptions.modelPath);
  log.info('Loaded model', modelOptions, 'tensors:', tf.engine().memory().numTensors, 'bytes:', tf.engine().memory().numBytes);

  // load image and get approprite tensor for it
  const inputSize = Object.values(model.modelSignature['inputs'])[0].tensorShape.dim[2].size;
  const imageFile = process.argv.length > 2 ? process.argv[2] : null;
  if (!imageFile || !fs.existsSync(imageFile)) {
    log.error('Specify a valid image file');
    process.exit();
  }
  const img = await loadImage(imageFile, inputSize);
  log.info('Loaded image:', img.fileName, 'inputShape:', img.inputShape, 'outputShape:', img.outputShape);

  // run actual prediction
  const res = model.predict(img.tensor);

  // process results
  const results = await processResults(res, inputSize, img.inputShape);

  // print results
  log.data('Results:', results);

  // save processed image
  await saveImage(img, results);
}

main();
