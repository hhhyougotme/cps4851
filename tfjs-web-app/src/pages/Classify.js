import React, { Component, Fragment } from 'react';
import {
  Alert, Button, Collapse, Container, Form, Spinner, ListGroup, Tabs, Tab
} from 'react-bootstrap';
import { FaCamera, FaChevronDown, FaChevronRight } from 'react-icons/fa';
import { openDB } from 'idb';
import Cropper  from 'react-cropper';
import * as tf from '@tensorflow/tfjs';
import LoadButton from '../components/LoadButton';
import { MODEL_CLASSES as MODEL_CLASSES_IMAGENET } from '../model/classes';
import config from '../config';
import './Classify.css';
import 'cropperjs/dist/cropper.css';

// Local model: set REACT_APP_MODEL_URL=/model/model.json in tfjs-web-app/.env
// 3-class fire/smoke: REACT_APP_NUM_CLASSES=3; Keras3 graph export also needs REACT_APP_MODEL_FORMAT=graph
// Default: CDN MobileNet v1 (ImageNet) when env unset, so empty public/model does not hang on Loading.
const MODEL_PATH =
  process.env.REACT_APP_MODEL_URL ||
  'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json';
const MODEL_FORMAT = (process.env.REACT_APP_MODEL_FORMAT || 'layers').toLowerCase();
const USE_GRAPH_MODEL = MODEL_FORMAT === 'graph';
const loadModelFromPath = (url) =>
  USE_GRAPH_MODEL ? tf.loadGraphModel(url) : tf.loadLayersModel(url);
const NUM_CLASSES_CFG = parseInt(process.env.REACT_APP_NUM_CLASSES || '1000', 10);
const MODEL_CLASSES_FIRE = ['Normal', 'Possible smoke', 'Possible fire'];
const HAZARD_CLASS_NAMES = new Set(['Possible smoke', 'Possible fire']);
const MODEL_CLASSES =
  NUM_CLASSES_CFG === 3 ? MODEL_CLASSES_FIRE : MODEL_CLASSES_IMAGENET;
const IMAGE_SIZE = 224;
const CANVAS_SIZE = 224;
const TOPK_PREDICTIONS = Math.min(5, MODEL_CLASSES.length);

const INDEXEDDB_DB = 'tensorflowjs';
const INDEXEDDB_STORE = 'model_info_store';
const INDEXEDDB_KEY = 'web-model';

/** Live tab: interval between inferences (ms). Lower = smoother but heavier CPU/GPU. */
const LIVE_INTERVAL_MS = 450;
/** Consecutive frames above hazard rule before showing alert (reduces flicker). */
const LIVE_HAZARD_STREAK = 3;

/**
 * Class to handle the rendering of the Classify page.
 * @extends React.Component
 */
export default class Classify extends Component {

  constructor(props) {
    super(props);

    this.webcam = null;
    this.model = null;
    this.modelLastUpdated = null;
    this.liveStream = null;
    this.liveMonitorTimer = null;
    this.liveHazardStreak = 0;
    this.liveFrameBusy = false;

    this.state = {
      modelLoaded: false,
      filename: '',
      isModelLoading: false,
      isClassifying: false,
      predictions: [],
      photoSettingsOpen: true,
      modelUpdateAvailable: false,
      showModelUpdateAlert: false,
      showModelUpdateSuccess: false,
      isDownloadingModel: false,
      hazardAlert: false,
      hazardMessage: '',
      inferMs: null,
      liveMonitoring: false
    };
  }

  async componentDidMount() {
    if (USE_GRAPH_MODEL) {
      this.model = await loadModelFromPath(MODEL_PATH);
    }
    else if (('indexedDB' in window)) {
      try {
        this.model = await tf.loadLayersModel('indexeddb://' + INDEXEDDB_KEY);

        // Safe to assume tensorflowjs database and related object store exists.
        // Get the date when the model was saved.
        try {
          const db = await openDB(INDEXEDDB_DB, 1, );
          const item = await db.transaction(INDEXEDDB_STORE)
                               .objectStore(INDEXEDDB_STORE)
                               .get(INDEXEDDB_KEY);
          const dateSaved = new Date(item.modelArtifactsInfo.dateSaved);
          await this.getModelInfo();
          console.log(this.modelLastUpdated);
          if (!this.modelLastUpdated  || dateSaved >= new Date(this.modelLastUpdated).getTime()) {
            console.log('Using saved model');
          }
          else {
            this.setState({
              modelUpdateAvailable: true,
              showModelUpdateAlert: true,
            });
          }

        }
        catch (error) {
          console.warn(error);
          console.warn('Could not retrieve when model was saved.');
        }

      }
      // If error here, assume that the object store doesn't exist and the model currently isn't
      // saved in IndexedDB.
      catch (error) {
        console.log('Not found in IndexedDB. Loading and saving...');
        console.log(error);
        this.model = await tf.loadLayersModel(MODEL_PATH);
        await this.model.save('indexeddb://' + INDEXEDDB_KEY);
      }
    }
    // If no IndexedDB, then just download like normal.
    else {
      console.warn('IndexedDB not supported.');
      this.model = await tf.loadLayersModel(MODEL_PATH);
    }

    this.setState({ modelLoaded: true });
    this.initWebcam();

    // Warm up model.
    let prediction = tf.tidy(() => this.model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])));
    prediction.dispose();
  }

  async componentWillUnmount() {
    this.stopLiveMonitor();
    if (this.webcam) {
      this.webcam.stop();
    }

    // Attempt to dispose of the model.
    try {
      this.model.dispose();
    }
    catch (e) {
      // Assume model is not loaded or already disposed.
    }
  }

  initWebcam = async () => {
    try {
      this.webcam = await tf.data.webcam(
        this.refs.webcam,
        {resizeWidth: CANVAS_SIZE, resizeHeight: CANVAS_SIZE, facingMode: 'environment'}
      );
    }
    catch (e) {
      this.refs.noWebcam.style.display = 'block';
    }
  }

  startWebcam = async () => {
    if (this.webcam) {
      this.webcam.start();
    }
  }

  stopWebcam = async () => {
    if (this.webcam) {
      this.webcam.stop();
    }
  }

  stopLiveMonitor = () => {
    if (this.liveMonitorTimer != null) {
      clearInterval(this.liveMonitorTimer);
      this.liveMonitorTimer = null;
    }
    if (this.liveStream) {
      this.liveStream.getTracks().forEach((t) => t.stop());
      this.liveStream = null;
    }
    const v = this.refs.liveVideo;
    if (v && v.srcObject) {
      v.srcObject = null;
    }
    this.liveHazardStreak = 0;
    this.liveFrameBusy = false;
    if (this.state.liveMonitoring) {
      this.setState({
        liveMonitoring: false,
        predictions: [],
        hazardAlert: false,
        hazardMessage: '',
        inferMs: null,
      });
    }
  };

  startLiveMonitor = async () => {
    if (this.state.liveMonitoring) {
      return;
    }
    const video = this.refs.liveVideo;
    if (!video) {
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 448 }, height: { ideal: 448 } },
        audio: false,
      });
      this.liveStream = stream;
      video.srcObject = stream;
      await video.play();
      this.liveHazardStreak = 0;
      this.setState({ liveMonitoring: true, hazardAlert: false, hazardMessage: '' });
      this.liveMonitorTimer = setInterval(() => {
        this.runLiveFrame();
      }, LIVE_INTERVAL_MS);
    }
    catch (e) {
      console.warn(e);
      this.setState({ liveMonitoring: false });
    }
  };

  runLiveFrame = async () => {
    if (!this.state.liveMonitoring || !this.model || this.liveFrameBusy) {
      return;
    }
    const video = this.refs.liveVideo;
    if (!video || video.readyState < 2) {
      return;
    }
    let img;
    let resized;
    let imageData;
    let logits;
    this.liveFrameBusy = true;
    try {
      img = tf.browser.fromPixels(video);
      resized = tf.image.resizeBilinear(img, [IMAGE_SIZE, IMAGE_SIZE]);
      imageData = await this.processImage(resized);
      const t0 = performance.now();
      logits = this.model.predict(imageData);
      const probabilities = await logits.data();
      const t1 = performance.now();
      const preds = await this.getTopKClasses(probabilities, TOPK_PREDICTIONS);
      const { hazardAlert: frameHazard, hazardMessage } = this.computeHazard(preds);

      if (NUM_CLASSES_CFG === 3 && frameHazard) {
        this.liveHazardStreak += 1;
      }
      else {
        this.liveHazardStreak = 0;
      }
      const hazardConfirmed =
        NUM_CLASSES_CFG === 3 &&
        this.liveHazardStreak >= LIVE_HAZARD_STREAK &&
        frameHazard;

      const ctx = this.refs.canvas && this.refs.canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(video, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
      }

      this.setState({
        predictions: preds,
        isClassifying: false,
        inferMs: (t1 - t0).toFixed(1),
        hazardAlert: hazardConfirmed,
        hazardMessage: hazardConfirmed ? hazardMessage : '',
      });
    }
    catch (e) {
      console.warn(e);
    }
    finally {
      if (img) {
        img.dispose();
      }
      if (resized) {
        resized.dispose();
      }
      if (imageData) {
        imageData.dispose();
      }
      if (logits) {
        logits.dispose();
      }
      this.liveFrameBusy = false;
    }
  };

  getModelInfo = async () => {
    await fetch(`${config.API_ENDPOINT}/model_info`, {
      method: 'GET',
    })
    .then(async (response) => {
      await response.json().then((data) => {
        this.modelLastUpdated = data.last_updated;
      })
      .catch((err) => {
        console.log('Unable to get parse model info.');
      });
    })
    .catch((err) => {
      console.log('Unable to get model info');
    });
  }

  updateModel = async () => {
    // Get the latest model from the server and refresh the one saved in IndexedDB.
    console.log('Updating the model: ' + INDEXEDDB_KEY);
    this.setState({ isDownloadingModel: true });
    if (this.model) {
      try {
        this.model.dispose();
      }
      catch (e) {
        // ignore
      }
    }
    this.model = await loadModelFromPath(MODEL_PATH);
    if (!USE_GRAPH_MODEL) {
      await this.model.save('indexeddb://' + INDEXEDDB_KEY);
    }
    this.setState({
      isDownloadingModel: false,
      modelUpdateAvailable: false,
      showModelUpdateAlert: false,
      showModelUpdateSuccess: true
    });
  }

  classifyLocalImage = async () => {
    this.setState({ isClassifying: true, hazardAlert: false, inferMs: null });

    const croppedCanvas = this.refs.cropper.getCroppedCanvas();
    const image = tf.tidy( () => tf.browser.fromPixels(croppedCanvas).toFloat());

    // Process and resize image before passing in to model.
    const imageData = await this.processImage(image);
    const resizedImage = tf.image.resizeBilinear(imageData, [IMAGE_SIZE, IMAGE_SIZE]);

    const t0 = performance.now();
    const logits = this.model.predict(resizedImage);
    const probabilities = await logits.data();
    const t1 = performance.now();
    const preds = await this.getTopKClasses(probabilities, TOPK_PREDICTIONS);
    const { hazardAlert, hazardMessage } = this.computeHazard(preds);
    const inferMs = (t1 - t0).toFixed(1);

    this.setState({
      predictions: preds,
      isClassifying: false,
      photoSettingsOpen: !this.state.photoSettingsOpen,
      hazardAlert,
      hazardMessage,
      inferMs
    });

    // Draw thumbnail to UI.
    const context = this.refs.canvas.getContext('2d');
    const ratioX = CANVAS_SIZE / croppedCanvas.width;
    const ratioY = CANVAS_SIZE / croppedCanvas.height;
    const ratio = Math.min(ratioX, ratioY);
    context.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    context.drawImage(croppedCanvas, 0, 0,
                      croppedCanvas.width * ratio, croppedCanvas.height * ratio);

    // Dispose of tensors we are finished with.
    image.dispose();
    imageData.dispose();
    resizedImage.dispose();
    logits.dispose();
  }

  classifyWebcamImage = async () => {
    this.setState({ isClassifying: true, hazardAlert: false, inferMs: null });

    const imageCapture = await this.webcam.capture();

    const resized = tf.image.resizeBilinear(imageCapture, [IMAGE_SIZE, IMAGE_SIZE]);
    const imageData = await this.processImage(resized);
    const t0 = performance.now();
    const logits = this.model.predict(imageData);
    const probabilities = await logits.data();
    const t1 = performance.now();
    const preds = await this.getTopKClasses(probabilities, TOPK_PREDICTIONS);
    const { hazardAlert, hazardMessage } = this.computeHazard(preds);
    const inferMs = (t1 - t0).toFixed(1);

    this.setState({
      predictions: preds,
      isClassifying: false,
      photoSettingsOpen: !this.state.photoSettingsOpen,
      hazardAlert,
      hazardMessage,
      inferMs
    });

    // Draw thumbnail to UI.
    const tensorData = tf.tidy(() => imageCapture.toFloat().div(255));
    await tf.browser.toPixels(tensorData, this.refs.canvas);

    // Dispose of tensors we are finished with.
    resized.dispose();
    imageCapture.dispose();
    imageData.dispose();
    logits.dispose();
    tensorData.dispose();
  }

  processImage = async (image) => {
    return tf.tidy(() => image.expandDims(0).toFloat().div(127).sub(1));
  }

  computeHazard(preds) {
    if (NUM_CLASSES_CFG !== 3 || !preds || preds.length === 0) {
      return { hazardAlert: false, hazardMessage: '' };
    }
    const top = preds[0];
    const p = parseFloat(top.probability);
    if (HAZARD_CLASS_NAMES.has(top.className) && p >= 55) {
      return {
        hazardAlert: true,
        hazardMessage:
          `Alert: model predicts "${top.className}" (${top.probability}% confidence). ` +
          'Please verify manually. Demo only — not for real fire-alarm use.'
      };
    }
    return { hazardAlert: false, hazardMessage: '' };
  }

  /**
   * Computes the probabilities of the topK classes given logits by computing
   * softmax to get probabilities and then sorting the probabilities.
   * @param logits Tensor representing the logits from MobileNet.
   * @param topK The number of top predictions to show.
   */
  getTopKClasses = async (values, topK) => {
    const valuesAndIndices = [];
    for (let i = 0; i < values.length; i++) {
      valuesAndIndices.push({value: values[i], index: i});
    }
    valuesAndIndices.sort((a, b) => {
      return b.value - a.value;
    });
    const topkValues = new Float32Array(topK);
    const topkIndices = new Int32Array(topK);
    for (let i = 0; i < topK; i++) {
      topkValues[i] = valuesAndIndices[i].value;
      topkIndices[i] = valuesAndIndices[i].index;
    }

    const topClassesAndProbs = [];
    for (let i = 0; i < topkIndices.length; i++) {
      topClassesAndProbs.push({
        className: MODEL_CLASSES[topkIndices[i]],
        probability: (topkValues[i] * 100).toFixed(2)
      });
    }
    return topClassesAndProbs;
  }

  handlePanelClick = event => {
    this.setState({ photoSettingsOpen: !this.state.photoSettingsOpen });
  }

  handleFileChange = event => {
    if (event.target.files && event.target.files.length > 0) {
      this.setState({
        file: URL.createObjectURL(event.target.files[0]),
        filename: event.target.files[0].name
      });
    }
  }

  handleTabSelect = activeKey => {
    switch(activeKey) {
      case 'camera':
        this.stopLiveMonitor();
        this.startWebcam();
        break;
      case 'localfile':
        this.stopLiveMonitor();
        this.setState({ filename: null, file: null });
        this.stopWebcam();
        break;
      case 'live':
        this.stopWebcam();
        break;
      default:
    }
  }

  render() {
    return (
      <div className="Classify container">

      { !this.state.modelLoaded &&
        <Fragment>
          <Spinner animation="border" role="status">
            <span className="sr-only">Loading...</span>
          </Spinner>
          {' '}<span className="loading-model-text">Loading Model</span>
        </Fragment>
      }

      { this.state.modelLoaded &&
        <Fragment>
        <Button
          onClick={this.handlePanelClick}
          className="classify-panel-header"
          aria-controls="photo-selection-pane"
          aria-expanded={this.state.photoSettingsOpen}
          >
          Take or Select a Photo to Classify
            <span className='panel-arrow'>
            { this.state.photoSettingsOpen
              ? <FaChevronDown />
              : <FaChevronRight />
            }
            </span>
          </Button>
          <Collapse in={this.state.photoSettingsOpen}>
            <div id="photo-selection-pane">
            { this.state.modelUpdateAvailable && this.state.showModelUpdateAlert &&
                <Container>
                  <Alert
                    variant="info"
                    show={this.state.modelUpdateAvailable && this.state.showModelUpdateAlert}
                    onClose={() => this.setState({ showModelUpdateAlert: false})}
                    dismissible>
                      An update for the <strong>{this.state.modelType}</strong> model is available.
                      <div className="d-flex justify-content-center pt-1">
                        {!this.state.isDownloadingModel &&
                          <Button onClick={this.updateModel}
                                  variant="outline-info">
                            Update
                          </Button>
                        }
                        {this.state.isDownloadingModel &&
                          <div>
                            <Spinner animation="border" role="status" size="sm">
                              <span className="sr-only">Downloading...</span>
                            </Spinner>
                            {' '}<strong>Downloading...</strong>
                          </div>
                        }
                      </div>
                  </Alert>
                </Container>
              }
              {this.state.showModelUpdateSuccess &&
                <Container>
                  <Alert variant="success"
                         onClose={() => this.setState({ showModelUpdateSuccess: false})}
                         dismissible>
                    The <strong>{this.state.modelType}</strong> model has been updated!
                  </Alert>
                </Container>
              }
            <Tabs defaultActiveKey="localfile" id="input-options" onSelect={this.handleTabSelect}
                  className="justify-content-center">
              <Tab eventKey="camera" title="Take Photo">
                <div id="no-webcam" ref="noWebcam">
                  <span className="camera-icon"><FaCamera /></span>
                  No camera found. <br />
                  Please use a device with a camera, or upload an image instead.
                </div>
                <div className="webcam-box-outer">
                  <div className="webcam-box-inner">
                    <video ref="webcam" autoPlay playsInline muted id="webcam"
                           width="448" height="448">
                    </video>
                  </div>
                </div>
                <div className="button-container">
                  <LoadButton
                    variant="primary"
                    size="lg"
                    onClick={this.classifyWebcamImage}
                    isLoading={this.state.isClassifying}
                    text="Classify"
                    loadingText="Classifying..."
                  />
                </div>
              </Tab>
              <Tab eventKey="localfile" title="Select Local File">
                <Form.Group controlId="file">
                  <Form.Label>Select Image File</Form.Label><br />
                  <Form.Label className="imagelabel">
                    {this.state.filename ? this.state.filename : 'Browse...'}
                  </Form.Label>
                  <Form.Control
                    onChange={this.handleFileChange}
                    type="file"
                    accept="image/*"
                    className="imagefile" />
                </Form.Group>
                { this.state.file &&
                  <Fragment>
                    <div id="local-image">
                      <Cropper
                        ref="cropper"
                        src={this.state.file}
                        style={{height: 400, width: '100%'}}
                        guides={true}
                        aspectRatio={1 / 1}
                        viewMode={2}
                      />
                    </div>
                    <div className="button-container">
                      <LoadButton
                        variant="primary"
                        size="lg"
                        disabled={!this.state.filename}
                        onClick={this.classifyLocalImage}
                        isLoading={this.state.isClassifying}
                        text="Classify"
                        loadingText="Classifying..."
                      />
                    </div>
                  </Fragment>
                }
              </Tab>
              <Tab eventKey="live" title="Live Monitor">
                <p className="text-muted small">
                  Continuous preview; runs inference about every {LIVE_INTERVAL_MS} ms. In 3-class mode,
                  an alert appears only after {LIVE_HAZARD_STREAK} consecutive frames meet the same
                  smoke/fire rule as single-shot mode (≥55% on top class). Same trained weights—no
                  retraining.
                </p>
                <div className="webcam-box-outer">
                  <div className="webcam-box-inner">
                    <video
                      ref="liveVideo"
                      autoPlay
                      playsInline
                      muted
                      id="live-webcam"
                      width="448"
                      height="448"
                    />
                  </div>
                </div>
                <div className="button-container">
                  {!this.state.liveMonitoring ? (
                    <Button variant="success" size="lg" onClick={this.startLiveMonitor}>
                      Start live monitoring
                    </Button>
                  ) : (
                    <Button variant="danger" size="lg" onClick={this.stopLiveMonitor}>
                      Stop live monitoring
                    </Button>
                  )}
                </div>
              </Tab>
            </Tabs>
            </div>
          </Collapse>
          { (this.state.predictions.length > 0 || this.state.liveMonitoring) &&
            <div className="classification-results">
              <h3>Predictions</h3>
              {this.state.hazardAlert &&
                <Alert variant="danger" className="mt-2">
                  {this.state.hazardMessage}
                </Alert>
              }
              {this.state.inferMs != null &&
                <p className="text-muted small mb-2">
                  Inference time (on-device TF.js): {this.state.inferMs} ms
                </p>
              }
              <canvas ref="canvas" width={CANVAS_SIZE} height={CANVAS_SIZE} />
              <br />
              <ListGroup>
              {this.state.predictions.map((category) => {
                  return (
                    <ListGroup.Item key={category.className}>
                      <strong>{category.className}</strong> {category.probability}%</ListGroup.Item>
                  );
              })}
              </ListGroup>
            </div>
          }
          </Fragment>
        }
      </div>
    );
  }
}
