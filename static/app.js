const LOW_CONFIDENCE_THRESHOLD = 0.7;
const MAX_FILE_SIZE = 5 * 1024 * 1024;
const OP_LABELS = ['+', '-', '·', '÷'];
const DIGIT_LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];

const stations = Array.from(document.querySelectorAll('.station'));
const statusMessage = document.getElementById('statusMessage');
const resultLine = document.getElementById('resultLine');
const confidencePanel = document.getElementById('confidencePanel');
const toggleConfidenceBtn = document.getElementById('toggleConfidenceBtn');
const predictBtn = document.getElementById('predictBtn');
const resultCard = document.getElementById('resultCard');
const historyList = document.getElementById('historyList');
const themeToggle = document.getElementById('themeToggle');
const soundToggle = document.getElementById('soundToggle');
const copyResultBtn = document.getElementById('copyResultBtn');
const copyToast = document.getElementById('copyToast');
const neuralBg = document.getElementById('neural-bg');

const stationState = new Map();
let soundEnabled = false;
let history = [];

function setStatus(message, tone = '') {
  statusMessage.textContent = message;
  statusMessage.className = `status-message ${tone}`.trim();
}

function playTone(type) {
  if (!soundEnabled) {
    return;
  }
  try {
    const context = new (window.AudioContext || window.webkitAudioContext)();
    const osc = context.createOscillator();
    const gain = context.createGain();
    osc.type = 'sine';
    osc.frequency.value = type === 'success' ? 820 : type === 'error' ? 220 : 500;
    gain.gain.value = 0.035;
    osc.connect(gain);
    gain.connect(context.destination);
    osc.start();
    osc.stop(context.currentTime + 0.08);
  } catch (_) {
  }
}

function initNeuralBackground() {
  const ctx = neuralBg.getContext('2d');
  const particles = [];
  const count = 44;

  function resize() {
    neuralBg.width = window.innerWidth;
    neuralBg.height = window.innerHeight;
  }

  function spawn() {
    particles.length = 0;
    for (let i = 0; i < count; i += 1) {
      particles.push({
        x: Math.random() * neuralBg.width,
        y: Math.random() * neuralBg.height,
        vx: (Math.random() - 0.5) * 0.25,
        vy: (Math.random() - 0.5) * 0.25,
        r: Math.random() * 1.6 + 0.8,
      });
    }
  }

  function draw() {
    ctx.clearRect(0, 0, neuralBg.width, neuralBg.height);

    for (const p of particles) {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0 || p.x > neuralBg.width) p.vx *= -1;
      if (p.y < 0 || p.y > neuralBg.height) p.vy *= -1;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(6,182,212,0.45)';
      ctx.fill();
    }

    for (let i = 0; i < particles.length; i += 1) {
      for (let j = i + 1; j < particles.length; j += 1) {
        const a = particles[i];
        const b = particles[j];
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d < 120) {
          ctx.strokeStyle = `rgba(168,85,247,${(1 - d / 120) * 0.16})`;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }

    requestAnimationFrame(draw);
  }

  resize();
  spawn();
  draw();
  window.addEventListener('resize', () => {
    resize();
    spawn();
  });
}

function rasterizeForInference(sourceCanvas) {
  const sourceCtx = sourceCanvas.getContext('2d');
  const { width, height } = sourceCanvas;
  const imageData = sourceCtx.getImageData(0, 0, width, height);
  const pixels = imageData.data;
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = (y * width + x) * 4;
      const darkness = 255 - pixels[index];
      if (darkness > 12) {
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
  }

  const outputCanvas = document.createElement('canvas');
  outputCanvas.width = 28;
  outputCanvas.height = 28;
  const outputCtx = outputCanvas.getContext('2d');
  outputCtx.fillStyle = '#ffffff';
  outputCtx.fillRect(0, 0, 28, 28);
  outputCtx.imageSmoothingEnabled = true;

  if (maxX < minX || maxY < minY) {
    return outputCanvas;
  }

  const cropWidth = maxX - minX + 1;
  const cropHeight = maxY - minY + 1;
  const scale = Math.min(20 / cropWidth, 20 / cropHeight);
  const targetWidth = Math.max(1, Math.round(cropWidth * scale));
  const targetHeight = Math.max(1, Math.round(cropHeight * scale));
  const offsetX = Math.floor((28 - targetWidth) / 2);
  const offsetY = Math.floor((28 - targetHeight) / 2);

  outputCtx.drawImage(sourceCanvas, minX, minY, cropWidth, cropHeight, offsetX, offsetY, targetWidth, targetHeight);
  return outputCanvas;
}

function drawPreviewFromCanvas(sourceCanvas, previewCanvas) {
  const pctx = previewCanvas.getContext('2d');
  const normalizedCanvas = rasterizeForInference(sourceCanvas);
  pctx.clearRect(0, 0, 28, 28);
  pctx.imageSmoothingEnabled = false;
  pctx.drawImage(normalizedCanvas, 0, 0, 28, 28);
}

function updateStationVisualState(state, visual) {
  visual.classList.remove('awaiting-input', 'error-state', 'success-state');
  if (state === 'awaiting-input') {
    visual.classList.add('awaiting-input');
  }
  if (state === 'error') {
    visual.classList.add('error-state');
  }
  if (state === 'success') {
    visual.classList.add('success-state');
  }
}

function animateRecognizedText(el, value) {
  el.textContent = '...';
  let steps = 8;
  const timer = setInterval(() => {
    steps -= 1;
    el.textContent = steps > 0 ? String(Math.floor(Math.random() * 10)) : value;
    if (steps <= 0) {
      clearInterval(timer);
    }
  }, 28);
}

function createCanvasController(canvas, slider, cursor) {
  const ctx = canvas.getContext('2d');
  let drawing = false;
  let lastPoint = null;

  function resetCanvas() {
    ctx.save();
    ctx.globalCompositeOperation = 'source-over';
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.restore();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#000000';
    ctx.fillStyle = '#000000';
    ctx.shadowBlur = 0;
    ctx.lineWidth = Number(slider.value);
  }

  function getPoint(event) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;
    const usesPressure = event.pointerType === 'pen';
    const rawPressure = usesPressure && typeof event.pressure === 'number' ? event.pressure : 0;
    const pressure = rawPressure > 0 ? rawPressure : 1;
    return { x, y, pressure };
  }

  function moveCursor(event) {
    const rect = canvas.getBoundingClientRect();
    cursor.style.left = `${event.clientX - rect.left}px`;
    cursor.style.top = `${event.clientY - rect.top}px`;
    cursor.style.opacity = '1';
    cursor.style.width = `${slider.value}px`;
    cursor.style.height = `${slider.value}px`;
  }

  function stampPoint(point) {
    const radius = (Number(slider.value) * point.pressure) / 2;
    ctx.beginPath();
    ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
    ctx.fill();
  }

  function beginStroke(point) {
    stampPoint(point);
    lastPoint = point;
  }

  function continueStroke(point) {
    if (!lastPoint) {
      beginStroke(point);
      return;
    }
    ctx.lineWidth = Number(slider.value) * point.pressure;
    ctx.beginPath();
    ctx.moveTo(lastPoint.x, lastPoint.y);
    ctx.lineTo(point.x, point.y);
    ctx.stroke();
    stampPoint(point);
    lastPoint = point;
  }

  function start(event) {
    event.preventDefault();
    drawing = true;
    canvas.setPointerCapture(event.pointerId);
    beginStroke(getPoint(event));
    moveCursor(event);
  }

  function move(event) {
    moveCursor(event);
    if (!drawing) return;
    event.preventDefault();
    const points = typeof event.getCoalescedEvents === 'function' ? event.getCoalescedEvents() : [event];
    points.forEach((pointEvent) => continueStroke(getPoint(pointEvent)));
  }

  function stop(event) {
    if (!drawing) return;
    if (event) {
      continueStroke(getPoint(event));
    }
    drawing = false;
    if (event && typeof event.pointerId === 'number') {
      try { canvas.releasePointerCapture(event.pointerId); } catch (_) {}
    }
    lastPoint = null;
    cursor.style.opacity = '0';
  }

  slider.addEventListener('input', () => {
    cursor.style.width = `${slider.value}px`;
    cursor.style.height = `${slider.value}px`;
  });

  canvas.addEventListener('pointerdown', start);
  canvas.addEventListener('pointermove', move);
  canvas.addEventListener('pointerup', stop);
  canvas.addEventListener('pointercancel', stop);
  canvas.addEventListener('pointerleave', () => {
    if (!drawing) cursor.style.opacity = '0';
  });

  resetCanvas();

  return {
    resetCanvas,
    toDataURL: () => canvas.toDataURL('image/png'),
  };
}

function initStation(stationEl) {
  const key = stationEl.dataset.key;
  const kind = stationEl.dataset.kind;
  const tabs = stationEl.querySelector('.tabs');
  const tabButtons = Array.from(stationEl.querySelectorAll('.tab-btn'));
  const drawPanel = stationEl.querySelector('.draw-panel');
  const uploadPanel = stationEl.querySelector('.upload-panel');
  const canvas = stationEl.querySelector('canvas');
  const cursor = stationEl.querySelector('.brush-cursor');
  const slider = stationEl.querySelector('.brush-slider');
  const clearBtn = stationEl.querySelector('.clear-btn');
  const previewBtn = stationEl.querySelector('.preview-btn');
  const previewWrap = stationEl.querySelector('.preview-wrap');
  const previewCanvas = stationEl.querySelector('.preview-28');
  const fileInput = stationEl.querySelector('.file-input');
  const dropzone = stationEl.querySelector('.dropzone');
  const uploadPreview = stationEl.querySelector('.upload-preview');
  const recognizedValue = stationEl.querySelector('.recognized-value');
  const confidenceValue = stationEl.querySelector('.confidence-value');
  const confidenceFill = stationEl.querySelector('.confidence-fill');

  const canvasController = createCanvasController(canvas, slider, cursor);

  const state = {
    key,
    kind,
    mode: 'draw',
    uploadedDataUrl: null,
    canvas,
    previewCanvas,
    canvasController,
    recognizedValue,
    confidenceValue,
    confidenceFill,
    stationEl,
    lastPrediction: null,
  };
  stationState.set(key, state);

  function setMode(mode) {
    state.mode = mode;
    tabButtons.forEach((btn) => btn.classList.toggle('active', btn.dataset.mode === mode));
    drawPanel.classList.toggle('active', mode === 'draw');
    uploadPanel.classList.toggle('active', mode === 'upload');
    tabs.classList.toggle('upload-active', mode === 'upload');
  }

  tabButtons.forEach((btn) => {
    btn.addEventListener('click', () => setMode(btn.dataset.mode));
  });

  function setUploadPreview(dataUrl) {
    uploadPreview.innerHTML = '';
    uploadPreview.classList.remove('placeholder');
    uploadPreview.classList.add('scan');
    const img = document.createElement('img');
    img.src = dataUrl;
    img.alt = `${key} upload`;
    uploadPreview.appendChild(img);
    setTimeout(() => uploadPreview.classList.remove('scan'), 1200);
  }

  function validateFile(file) {
    if (!file) return 'No file selected';
    if (file.size > MAX_FILE_SIZE) return 'File too large. Please choose an image under 5MB.';
    if (!['image/png', 'image/jpeg'].includes(file.type)) return 'Invalid file type. Please upload PNG, JPG, or JPEG.';
    return null;
  }

  async function readFile(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  async function processFile(file) {
    const err = validateFile(file);
    if (err) {
      setStatus(err, 'error');
      updateStationVisualState('error', stationEl);
      return;
    }
    const dataUrl = await readFile(file);
    state.uploadedDataUrl = dataUrl;
    setUploadPreview(dataUrl);
    updateStationVisualState('success', stationEl);
    setStatus('Upload loaded.', 'success');
  }

  fileInput.addEventListener('change', async (event) => {
    const [file] = event.target.files;
    if (!file) return;
    await processFile(file);
  });

  dropzone.addEventListener('dragover', (event) => {
    event.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });

  dropzone.addEventListener('drop', async (event) => {
    event.preventDefault();
    dropzone.classList.remove('dragover');
    const file = event.dataTransfer.files[0];
    await processFile(file);
  });

  clearBtn.addEventListener('click', () => {
    stationEl.animate([{ opacity: 1 }, { opacity: 0.65 }, { opacity: 1 }], { duration: 220, easing: 'ease-out' });
    canvasController.resetCanvas();
    state.uploadedDataUrl = null;
    fileInput.value = '';
    uploadPreview.innerHTML = 'No image selected';
    uploadPreview.classList.add('placeholder');
    recognizedValue.textContent = '—';
    confidenceValue.textContent = '0%';
    confidenceFill.style.width = '0%';
    updateStationVisualState('awaiting-input', stationEl);
  });

  previewBtn.addEventListener('click', () => {
    previewWrap.classList.toggle('hidden');
    if (state.mode === 'draw') {
      drawPreviewFromCanvas(canvas, previewCanvas);
    } else if (state.uploadedDataUrl) {
      const img = new Image();
      img.onload = () => {
        const pctx = previewCanvas.getContext('2d');
        pctx.clearRect(0, 0, 28, 28);
        pctx.drawImage(img, 0, 0, 28, 28);
      };
      img.src = state.uploadedDataUrl;
    }
  });

  setMode('draw');
}

function getStationImage(state) {
  if (state.mode === 'draw') {
    const normalizedCanvas = rasterizeForInference(state.canvas);
    return normalizedCanvas.toDataURL('image/png');
  }
  return state.uploadedDataUrl;
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

function renderConfidenceGroups(groups) {
  confidencePanel.innerHTML = '';
  groups.forEach((group) => {
    const section = document.createElement('section');
    section.className = 'confidence-group';

    const title = document.createElement('h3');
    title.textContent = group.title;
    section.appendChild(title);

    group.labels.forEach((label, index) => {
      const row = document.createElement('div');
      row.className = 'confidence-row';
      if (index === group.predictedIndex) row.classList.add('predicted');

      const labelEl = document.createElement('div');
      labelEl.textContent = label;

      const bar = document.createElement('div');
      bar.className = 'confidence-bar';

      const fill = document.createElement('div');
      fill.className = 'confidence-fill-bar';
      fill.style.width = `${Math.max(2, group.probabilities[index] * 100)}%`;
      bar.appendChild(fill);

      const pct = document.createElement('div');
      pct.textContent = `${(group.probabilities[index] * 100).toFixed(1)}%`;

      row.append(labelEl, bar, pct);
      section.appendChild(row);
    });

    confidencePanel.appendChild(section);
  });
}

function updateStationPrediction(state, prediction, confidence) {
  animateRecognizedText(state.recognizedValue, String(prediction));
  state.confidenceValue.textContent = `${Math.round(confidence * 100)}%`;
  state.confidenceFill.style.width = `${Math.max(3, confidence * 100)}%`;
  updateStationVisualState(confidence < LOW_CONFIDENCE_THRESHOLD ? 'awaiting-input' : 'success', state.stationEl);
}

function updateHistory(equation) {
  history.unshift(equation);
  history = history.slice(0, 8);
  historyList.innerHTML = '';
  history.forEach((item) => {
    const li = document.createElement('li');
    li.textContent = item;
    historyList.appendChild(li);
  });
}

function typewriterResult(text) {
  resultLine.textContent = '';
  let i = 0;
  const tick = () => {
    if (i > text.length) return;
    resultLine.textContent = text.slice(0, i);
    i += 1;
    requestAnimationFrame(tick);
  };
  tick();
}

async function predictAndCalculate() {
  try {
    setStatus('Processing neural inference...', '');
    predictBtn.classList.add('processing');
    resultCard.classList.remove('error-flare', 'success-flare');

    const d1 = stationState.get('d1');
    const op = stationState.get('op');
    const d2 = stationState.get('d2');

    const d1Img = getStationImage(d1);
    const opImg = getStationImage(op);
    const d2Img = getStationImage(d2);

    if (!d1Img || !opImg || !d2Img) {
      [d1, op, d2].forEach((st) => {
        if (!getStationImage(st)) updateStationVisualState('awaiting-input', st.stationEl);
      });
      throw new Error('Please draw or upload an image');
    }

    const [pred1, predOp, pred2] = await Promise.all([
      postJson('/predict_digit', { image: d1Img }),
      postJson('/predict_operator', { image: opImg }),
      postJson('/predict_digit', { image: d2Img }),
    ]);

    updateStationPrediction(d1, pred1.digit, pred1.confidence);
    updateStationPrediction(op, predOp.operator, predOp.confidence);
    updateStationPrediction(d2, pred2.digit, pred2.confidence);

    const calc = await postJson('/calculate', { d1: pred1.digit, op: predOp.operator, d2: pred2.digit });

    const expression = `${pred1.digit} ${predOp.operator} ${pred2.digit} = ${calc.result}`;
    typewriterResult(`Recognized: ${expression}`);

    const warnings = [pred1.warning, predOp.warning, pred2.warning].filter(Boolean);
    const hasError = typeof calc.result === 'string' && calc.result.toLowerCase().includes('error');

    if (hasError) {
      setStatus(calc.result === 'Error: Cannot divide by zero' ? '∅ UNDEFINED' : calc.result, 'error');
      resultCard.classList.add('error-flare');
      playTone('error');
    } else {
      setStatus(warnings[0] || 'Computation complete.', warnings.length ? 'warning' : 'success');
      resultCard.classList.add('success-flare');
      playTone('success');
    }

    renderConfidenceGroups([
      { title: 'Digit 1', labels: DIGIT_LABELS, probabilities: pred1.probabilities, predictedIndex: pred1.digit },
      { title: 'Operator', labels: OP_LABELS, probabilities: predOp.probabilities, predictedIndex: OP_LABELS.indexOf(predOp.operator) },
      { title: 'Digit 2', labels: DIGIT_LABELS, probabilities: pred2.probabilities, predictedIndex: pred2.digit },
    ]);

    updateHistory(expression);
  } catch (error) {
    setStatus(error.message || 'Could not recognize symbol', 'error');
    resultCard.classList.add('error-flare');
    playTone('error');
  } finally {
    predictBtn.classList.remove('processing');
  }
}

predictBtn.addEventListener('click', predictAndCalculate);

toggleConfidenceBtn.addEventListener('click', () => {
  const hidden = confidencePanel.classList.toggle('hidden');
  toggleConfidenceBtn.textContent = hidden ? 'Show Neural Insights' : 'Hide Neural Insights';
});

document.addEventListener('keydown', (event) => {
  if (event.ctrlKey && event.key === 'Enter') {
    event.preventDefault();
    predictAndCalculate();
  }
});

themeToggle.addEventListener('click', () => {
  document.body.classList.toggle('light');
  const dark = !document.body.classList.contains('light');
  themeToggle.textContent = dark ? 'Dark Mode' : 'Light Mode';
  themeToggle.setAttribute('aria-pressed', String(dark));
});

soundToggle.addEventListener('click', () => {
  soundEnabled = !soundEnabled;
  soundToggle.textContent = `Sound: ${soundEnabled ? 'On' : 'Off'}`;
  soundToggle.setAttribute('aria-pressed', String(soundEnabled));
});

copyResultBtn.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(resultLine.textContent.replace('Recognized: ', ''));
    copyToast.classList.add('show');
    setTimeout(() => copyToast.classList.remove('show'), 1000);
  } catch (_) {
    setStatus('Copy failed.', 'warning');
  }
});

stations.forEach(initStation);
initNeuralBackground();
setStatus('Ready.', '');
