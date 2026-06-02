/**
 * Run TF.js graph model on image paths (one probability vector per line JSON stdout).
 * Usage (from tfjs-web-app): node ../scripts/evaluate_tfjs_subset.js <paths.txt> <model.json file URL>
 */
const fs = require('fs');
const path = require('path');

const tf = require('@tensorflow/tfjs');
const jpeg = require('jpeg-js');

const pathsFile = process.argv[2];
const modelUrl = process.argv[3];
const IMG = 224;

if (!pathsFile || !modelUrl) {
  console.error('Usage: node evaluate_tfjs_subset.js <paths.txt> <model.json URL>');
  process.exit(1);
}

const lines = fs.readFileSync(pathsFile, 'utf8').trim().split(/\r?\n/).filter(Boolean);

function loadAndPreprocess(filePath) {
  const buf = fs.readFileSync(filePath);
  const decoded = jpeg.decode(buf, { useTArray: true });
  const { width, height, data } = decoded;
  const canvas = tf.tidy(() => {
    let img = tf.tensor3d(data, [height, width, 4]).slice([0, 0, 0], [-1, -1, 3]);
    if (width !== IMG || height !== IMG) {
      img = tf.image.resizeBilinear(img, [IMG, IMG]);
    }
    const batched = img.expandDims(0).toFloat().div(127).sub(1);
    return batched;
  });
  return canvas;
}

(async () => {
  await tf.setBackend('cpu');
  await tf.ready();
  const model = await tf.loadGraphModel(modelUrl);
  for (const p of lines) {
    const input = loadAndPreprocess(p);
    const logits = tf.tidy(() => model.predict(input));
    const probs = await logits.data();
    tf.dispose([input, logits]);
    const arr = Array.from(probs).slice(0, 3);
    console.log(JSON.stringify(arr));
  }
  model.dispose();
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
