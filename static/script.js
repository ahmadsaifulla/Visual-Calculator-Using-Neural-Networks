const CANVAS_SIZE = 280;
const MODEL_SIZE = 28;
const BRUSH_WIDTH = 15;
const LOW_CONFIDENCE_THRESHOLD = 0.7;

const boxes = Array.from(document.querySelectorAll('.input-box'));
const calculateBtn = document.getElementById('calculateBtn');
const resetAllBtn = document.getElementById('resetAllBtn');
const equationDisplay = document.getElementById('equationDisplay');
const statusLine = document.getElementById('statusLine');

const state = new Map();

function setStatus(message) {
  statusLine.textContent = message;
}

function setEquation(text, isError = false) {
  equationDisplay.textContent = text;
  equationDisplay.classList.add('visible');
  equationDisplay.classList.toggle('error', isError);
}

function getConfidenceClass(confidence) {
  if (confidence > 0.9) return 'good';
  if (confidence >= LOW_CONFIDENCE_THRESHOLD) return 'warn';
  return 'bad';
}

function updateCalculateState() {
  const ready = Array.from(state.values()).every((entry) => entry.hasInput);
  calculateBtn.disabled = !ready;
  if (!ready) {
    setStatus('Waiting for all 3 inputs.');
  }
}

function resetPrediction(entry) {
  entry.predictionSymbol.textContent = '—';
  entry.predictionConfidence.textContent = 'Confidence: 0%';
  entry.predictionConfidence.className = 'prediction-confidence neutral';
  entry.predictionWarning.textContent = '';
  entry.prediction = null;
}

function clearEntry(entry) {
  entry.ctx.fillStyle = '#ffffff';
  entry.ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  entry.uploadedDataUrl = null;
  entry.fileInput.value = '';
  entry.uploadPreview.innerHTML = 'No image selected';
  entry.hasInput = false;
  resetPrediction(entry);
  updateCalculateState();
}

function switchMode(entry, mode) {
  entry.mode = mode;
  entry.drawPanel.classList.toggle('active', mode === 'draw');
  entry.uploadPanel.classList.toggle('active', mode === 'upload');
  clearEntry(entry);
}

function normalizeImage(source) {
  const sourceCanvas = document.createElement('canvas');
  sourceCanvas.width = CANVAS_SIZE;
  sourceCanvas.height = CANVAS_SIZE;
  const sourceCtx = sourceCanvas.getContext('2d');
  sourceCtx.fillStyle = '#ffffff';
  sourceCtx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  sourceCtx.drawImage(source, 0, 0, CANVAS_SIZE, CANVAS_SIZE);

  const sourceData = sourceCtx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  const pixels = sourceData.data;
  let sum = 0;
  let minX = CANVAS_SIZE;
  let minY = CANVAS_SIZE;
  let maxX = -1;
  let maxY = -1;

  for (let i = 0; i < pixels.length; i += 4) {
    const gray = Math.round(0.299 * pixels[i] + 0.587 * pixels[i + 1] + 0.114 * pixels[i + 2]);
    pixels[i] = gray;
    pixels[i + 1] = gray;
    pixels[i + 2] = gray;
    pixels[i + 3] = 255;
    sum += gray / 255;
  }

  const mean = sum / (CANVAS_SIZE * CANVAS_SIZE);
  if (mean < 0.5) {
    for (let i = 0; i < pixels.length; i += 4) {
      const value = 255 - pixels[i];
      pixels[i] = value;
      pixels[i + 1] = value;
      pixels[i + 2] = value;
    }
  }

  for (let y = 0; y < CANVAS_SIZE; y += 1) {
    for (let x = 0; x < CANVAS_SIZE; x += 1) {
      const index = (y * CANVAS_SIZE + x) * 4;
      if (pixels[index] < 245) {
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
  }

  sourceCtx.putImageData(sourceData, 0, 0);

  const modelCanvas = document.createElement('canvas');
  modelCanvas.width = MODEL_SIZE;
  modelCanvas.height = MODEL_SIZE;
  const modelCtx = modelCanvas.getContext('2d');
  modelCtx.fillStyle = '#ffffff';
  modelCtx.fillRect(0, 0, MODEL_SIZE, MODEL_SIZE);

  if (maxX >= minX && maxY >= minY) {
    const cropWidth = maxX - minX + 1;
    const cropHeight = maxY - minY + 1;
    const scale = Math.min(20 / cropWidth, 20 / cropHeight);
    const targetWidth = Math.max(1, Math.round(cropWidth * scale));
    const targetHeight = Math.max(1, Math.round(cropHeight * scale));
    const offsetX = Math.floor((MODEL_SIZE - targetWidth) / 2);
    const offsetY = Math.floor((MODEL_SIZE - targetHeight) / 2);
    modelCtx.imageSmoothingEnabled = false;
    modelCtx.drawImage(sourceCanvas, minX, minY, cropWidth, cropHeight, offsetX, offsetY, targetWidth, targetHeight);
  }

  return modelCanvas.toDataURL('image/png');
}

function loadImage(dataUrl) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = dataUrl;
  });
}

async function getEntryImage(entry) {
  if (entry.mode === 'draw') {
    return normalizeImage(entry.canvas);
  }
  if (!entry.uploadedDataUrl) {
    return null;
  }
  const image = await loadImage(entry.uploadedDataUrl);
  return normalizeImage(image);
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || 'Request failed');
  }
  return data;
}

function showPrediction(entry, prediction, confidence) {
  entry.predictionSymbol.textContent = String(prediction);
  entry.predictionConfidence.textContent = `Confidence: ${Math.round(confidence * 100)}%`;
  entry.predictionConfidence.className = `prediction-confidence ${getConfidenceClass(confidence)}`;
  entry.predictionWarning.textContent = confidence < LOW_CONFIDENCE_THRESHOLD ? 'Low confidence — try drawing clearer' : '';
  entry.prediction = prediction;
}

function setupCanvas(entry) {
  let drawing = false;
  let lastPoint = null;

  entry.ctx.fillStyle = '#ffffff';
  entry.ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  entry.ctx.strokeStyle = '#000000';
  entry.ctx.lineWidth = BRUSH_WIDTH;
  entry.ctx.lineCap = 'round';
  entry.ctx.lineJoin = 'round';

  function getPoint(event) {
    const rect = entry.canvas.getBoundingClientRect();
    const scaleX = entry.canvas.width / rect.width;
    const scaleY = entry.canvas.height / rect.height;
    return {
      x: (event.clientX - rect.left) * scaleX,
      y: (event.clientY - rect.top) * scaleY,
    };
  }

  function drawTo(point) {
    if (!lastPoint) {
      lastPoint = point;
      entry.ctx.beginPath();
      entry.ctx.arc(point.x, point.y, BRUSH_WIDTH / 2, 0, Math.PI * 2);
      entry.ctx.fillStyle = '#000000';
      entry.ctx.fill();
      return;
    }
    const midX = (lastPoint.x + point.x) / 2;
    const midY = (lastPoint.y + point.y) / 2;
    entry.ctx.beginPath();
    entry.ctx.moveTo(lastPoint.x, lastPoint.y);
    entry.ctx.quadraticCurveTo(lastPoint.x, lastPoint.y, midX, midY);
    entry.ctx.stroke();
    lastPoint = point;
  }

  entry.canvas.addEventListener('pointerdown', (event) => {
    event.preventDefault();
    drawing = true;
    entry.canvas.setPointerCapture(event.pointerId);
    drawTo(getPoint(event));
    entry.hasInput = true;
    updateCalculateState();
  });

  entry.canvas.addEventListener('pointermove', (event) => {
    if (!drawing) return;
    event.preventDefault();
    drawTo(getPoint(event));
  });

  function stop(event) {
    if (!drawing) return;
    drawing = false;
    lastPoint = null;
    try { entry.canvas.releasePointerCapture(event.pointerId); } catch (_) {}
  }

  entry.canvas.addEventListener('pointerup', stop);
  entry.canvas.addEventListener('pointercancel', stop);
}

function initBox(box) {
  const key = box.dataset.key;
  const modeSelect = box.querySelector('.mode-select');
  const drawPanel = box.querySelector('.draw-panel');
  const uploadPanel = box.querySelector('.upload-panel');
  const canvas = box.querySelector('canvas');
  const ctx = canvas.getContext('2d');
  const fileInput = box.querySelector('.file-input');
  const dropzone = box.querySelector('.dropzone');
  const uploadPreview = box.querySelector('.upload-preview');
  const clearBtn = box.querySelector('.clear-btn');
  const predictionSymbol = box.querySelector('.prediction-symbol');
  const predictionConfidence = box.querySelector('.prediction-confidence');
  const predictionWarning = box.querySelector('.prediction-warning');

  const entry = {
    key,
    type: box.dataset.type,
    mode: 'draw',
    drawPanel,
    uploadPanel,
    canvas,
    ctx,
    fileInput,
    dropzone,
    uploadPreview,
    clearBtn,
    predictionSymbol,
    predictionConfidence,
    predictionWarning,
    uploadedDataUrl: null,
    hasInput: false,
    prediction: null,
  };

  state.set(key, entry);
  setupCanvas(entry);

  modeSelect.addEventListener('change', () => switchMode(entry, modeSelect.value));

  clearBtn.addEventListener('click', () => clearEntry(entry));

  fileInput.addEventListener('change', async (event) => {
    const [file] = event.target.files;
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      entry.uploadedDataUrl = reader.result;
      entry.uploadPreview.innerHTML = `<img src="${reader.result}" alt="upload preview">`;
      entry.hasInput = true;
      updateCalculateState();
    };
    reader.readAsDataURL(file);
  });

  dropzone.addEventListener('dragover', (event) => {
    event.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });

  dropzone.addEventListener('drop', (event) => {
    event.preventDefault();
    dropzone.classList.remove('dragover');
    const [file] = event.dataTransfer.files;
    if (!file) return;
    fileInput.files = event.dataTransfer.files;
    fileInput.dispatchEvent(new Event('change'));
  });

  clearEntry(entry);
}

async function predictAll() {
  setStatus('Running predictions...');
  const d1 = state.get('d1');
  const op = state.get('op');
  const d2 = state.get('d2');

  const [d1Image, opImage, d2Image] = await Promise.all([
    getEntryImage(d1),
    getEntryImage(op),
    getEntryImage(d2),
  ]);

  const [digit1, operator, digit2] = await Promise.all([
    postJson('/predict_digit', { image: d1Image }),
    postJson('/predict_operator', { image: opImage }),
    postJson('/predict_digit', { image: d2Image }),
  ]);

  showPrediction(d1, digit1.digit, digit1.confidence);
  showPrediction(op, operator.operator, operator.confidence);
  showPrediction(d2, digit2.digit, digit2.confidence);

  return {
    d1: digit1.digit,
    op: operator.operator,
    d2: digit2.digit,
  };
}

async function calculateEquation() {
  try {
    calculateBtn.disabled = true;
    setStatus('Calculating...');
    const prediction = await predictAll();
    const result = await postJson('/calculate', prediction);
    setEquation(`${result.equation} = ${result.result}`, typeof result.result === 'string' && result.result.includes('Error'));
    setStatus('Calculation complete.');
  } catch (error) {
    setStatus(error.message || 'Calculation failed.');
    setEquation(error.message || 'Calculation failed.', true);
  } finally {
    updateCalculateState();
  }
}

function resetAll() {
  state.forEach((entry) => clearEntry(entry));
  setEquation('—');
  equationDisplay.classList.remove('visible', 'error');
  setStatus('Waiting for all 3 inputs.');
}

boxes.forEach(initBox);
calculateBtn.addEventListener('click', calculateEquation);
resetAllBtn.addEventListener('click', resetAll);
updateCalculateState();
