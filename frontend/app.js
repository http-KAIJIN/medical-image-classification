const API_BASE_URL = 'http://127.0.0.1:8000';
const ACCEPTED_EXTENSIONS = ['jpg', 'jpeg', 'png'];

const elements = {
  apiStatus: document.getElementById('apiStatus'),
  dropZone: document.getElementById('dropZone'),
  fileInput: document.getElementById('fileInput'),
  predictBtn: document.getElementById('predictBtn'),
  gradcamBtn: document.getElementById('gradcamBtn'),
  resetBtn: document.getElementById('resetBtn'),
  messageBox: document.getElementById('messageBox'),
  previewFrame: document.getElementById('previewFrame'),
  previewImage: document.getElementById('previewImage'),
  originalFrame: document.getElementById('originalFrame'),
  originalComparison: document.getElementById('originalComparison'),
  gradcamFrame: document.getElementById('gradcamFrame'),
  gradcamImage: document.getElementById('gradcamImage'),
  fileMeta: document.getElementById('fileMeta'),
  resultEmpty: document.getElementById('resultEmpty'),
  resultPanel: document.getElementById('resultPanel'),
  predictionBadge: document.getElementById('predictionBadge'),
  confidenceValue: document.getElementById('confidenceValue'),
  probabilityValue: document.getElementById('probabilityValue'),
  thresholdValue: document.getElementById('thresholdValue'),
  loadingBackdrop: document.getElementById('loadingBackdrop'),
  loadingText: document.getElementById('loadingText'),
};

let selectedFile = null;
let previewUrl = null;

function formatPercent(value) {
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function setLoading(isLoading, text = 'Processing image...') {
  elements.loadingText.textContent = text;
  elements.loadingBackdrop.hidden = !isLoading;
  elements.predictBtn.disabled = isLoading || !selectedFile;
  elements.gradcamBtn.disabled = isLoading || !selectedFile;
}

function showMessage(message, type = 'info') {
  elements.messageBox.textContent = message;
  elements.messageBox.className = `message ${type}`;
  elements.messageBox.hidden = false;
}

function clearMessage() {
  elements.messageBox.hidden = true;
  elements.messageBox.textContent = '';
}

function validateFile(file) {
  if (!file) {
    return 'Please select an image first.';
  }
  const extension = file.name.split('.').pop().toLowerCase();
  if (!ACCEPTED_EXTENSIONS.includes(extension)) {
    return 'Invalid file type. Please upload a JPG, JPEG, or PNG image.';
  }
  if (file.size === 0) {
    return 'The selected file is empty. Please choose a valid chest X-ray image.';
  }
  return null;
}

function setImage(frame, imageElement, src) {
  imageElement.src = src;
  imageElement.hidden = false;
  const placeholder = frame.querySelector('span');
  if (placeholder) placeholder.hidden = true;
  frame.classList.remove('empty');
}

function clearImage(frame, imageElement) {
  imageElement.removeAttribute('src');
  imageElement.hidden = true;
  const placeholder = frame.querySelector('span');
  if (placeholder) placeholder.hidden = false;
  frame.classList.add('empty');
}

function handleFile(file) {
  clearMessage();
  const error = validateFile(file);
  if (error) {
    showMessage(error, 'error');
    return;
  }

  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);

  setImage(elements.previewFrame, elements.previewImage, previewUrl);
  setImage(elements.originalFrame, elements.originalComparison, previewUrl);
  clearImage(elements.gradcamFrame, elements.gradcamImage);

  elements.fileMeta.textContent = `${file.name} • ${(file.size / 1024).toFixed(1)} KB`;
  elements.predictBtn.disabled = false;
  elements.gradcamBtn.disabled = false;
  elements.resetBtn.disabled = false;
  resetPredictionDisplay();
}

function resetPredictionDisplay() {
  elements.resultEmpty.hidden = false;
  elements.resultPanel.hidden = true;
  elements.predictionBadge.textContent = '--';
  elements.predictionBadge.className = 'prediction-badge';
}

function displayPrediction(result) {
  elements.resultEmpty.hidden = true;
  elements.resultPanel.hidden = false;
  elements.predictionBadge.textContent = result.prediction;
  elements.predictionBadge.className = `prediction-badge ${result.prediction === 'PNEUMONIA' ? 'pneumonia' : 'normal'}`;
  elements.confidenceValue.textContent = formatPercent(result.confidence);
  elements.probabilityValue.textContent = formatPercent(result.probability);
  elements.thresholdValue.textContent = Number(result.threshold).toFixed(2);
}

function makeFormData() {
  const formData = new FormData();
  formData.append('file', selectedFile);
  return formData;
}

async function parseApiResponse(response, fallbackMessage) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || fallbackMessage);
  }
  return data;
}

async function runPrediction() {
  const error = validateFile(selectedFile);
  if (error) {
    showMessage(error, 'error');
    return;
  }

  setLoading(true, 'Running model prediction...');
  clearMessage();
  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      body: makeFormData(),
    });
    const result = await parseApiResponse(response, 'Prediction failed. Please try again.');
    displayPrediction(result);
    showMessage('Prediction completed successfully.', 'info');
  } catch (error) {
    showMessage(error.message || 'API unavailable. Confirm the backend is running.', 'error');
  } finally {
    setLoading(false);
  }
}

async function runGradcam() {
  const error = validateFile(selectedFile);
  if (error) {
    showMessage(error, 'error');
    return;
  }

  setLoading(true, 'Generating Grad-CAM visualization...');
  clearMessage();
  try {
    const response = await fetch(`${API_BASE_URL}/gradcam`, {
      method: 'POST',
      body: makeFormData(),
    });
    const result = await parseApiResponse(response, 'Grad-CAM generation failed. Please try again.');
    setImage(elements.gradcamFrame, elements.gradcamImage, `${API_BASE_URL}${result.gradcam_image}`);
    showMessage(`Grad-CAM generated. Prediction: ${result.prediction}, confidence: ${formatPercent(result.confidence)}.`, 'info');
  } catch (error) {
    showMessage(error.message || 'API unavailable. Confirm the backend is running.', 'error');
  } finally {
    setLoading(false);
  }
}

function resetInterface() {
  selectedFile = null;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  elements.fileInput.value = '';
  elements.fileMeta.textContent = 'Choose an image to begin.';
  elements.predictBtn.disabled = true;
  elements.gradcamBtn.disabled = true;
  elements.resetBtn.disabled = true;
  clearImage(elements.previewFrame, elements.previewImage);
  clearImage(elements.originalFrame, elements.originalComparison);
  clearImage(elements.gradcamFrame, elements.gradcamImage);
  resetPredictionDisplay();
  clearMessage();
}

async function checkApiStatus() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    const data = await parseApiResponse(response, 'Health check failed.');
    elements.apiStatus.textContent = `${data.status.toUpperCase()} • ${data.model} v${data.version}`;
    elements.apiStatus.className = 'status-pill ok';
  } catch (error) {
    elements.apiStatus.textContent = 'API unavailable';
    elements.apiStatus.className = 'status-pill error';
  }
}

elements.fileInput.addEventListener('change', (event) => {
  handleFile(event.target.files[0]);
});

elements.dropZone.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    elements.fileInput.click();
  }
});

['dragenter', 'dragover'].forEach((eventName) => {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.add('dragover');
  });
});

['dragleave', 'drop'].forEach((eventName) => {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.remove('dragover');
  });
});

elements.dropZone.addEventListener('drop', (event) => {
  handleFile(event.dataTransfer.files[0]);
});

elements.predictBtn.addEventListener('click', runPrediction);
elements.gradcamBtn.addEventListener('click', runGradcam);
elements.resetBtn.addEventListener('click', resetInterface);

checkApiStatus();
