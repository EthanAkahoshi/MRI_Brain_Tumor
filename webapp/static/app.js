const MAX_BYTES = 10 * 1024 * 1024;
const UNSURE_BELOW = 0.75;

const $ = (id) => document.getElementById(id);
const dropzone = $('dropzone');
const fileInput = $('file-input');
const errorBox = $('error');
const loading = $('loading');
const results = $('results');
const heatmap = $('heatmap');
const opacity = $('opacity');

const percent = (value) => `${(value * 100).toFixed(1)}%`;

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function setOpacity() {
  const visible = $('heatmap-toggle').checked;
  heatmap.style.opacity = visible ? opacity.value / 100 : 0;
  $('opacity-value').textContent = `${opacity.value}%`;
  opacity.disabled = !visible;
}

function showResult(data) {
  $('scan').src = data.image;
  heatmap.src = data.heatmap;
  $('model-view').src = data.model_view;

  const prediction = $('prediction');
  prediction.textContent = data.prediction;
  prediction.className = `prediction ${data.prediction === 'Tumor' ? 'tumor' : 'clear'}`;

  $('confidence-bar').style.width = percent(data.confidence);
  $('confidence').textContent = `${percent(data.confidence)} confidence`;

  $('probabilities').replaceChildren(...Object.entries(data.probabilities).flatMap(([name, p]) => {
    const dt = document.createElement('dt');
    const dd = document.createElement('dd');
    dt.textContent = name;
    dd.textContent = percent(p);
    return [dt, dd];
  }));

  $('uncertain').hidden = data.confidence >= UNSURE_BELOW;

  $('warnings').replaceChildren(...data.warnings.map((text) => {
    const li = document.createElement('li');
    li.textContent = text;
    return li;
  }));

  setOpacity();
  results.hidden = false;
  results.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

async function analyse(file) {
  errorBox.hidden = true;

  if (!['image/jpeg', 'image/png'].includes(file.type)) {
    showError('Please choose a JPEG or PNG image.');
    return;
  }
  if (file.size > MAX_BYTES) {
    showError('That image is larger than 10 MB.');
    return;
  }

  results.hidden = true;
  loading.hidden = false;

  try {
    const body = new FormData();
    body.append('file', file);
    const response = await fetch('/api/predict', { method: 'POST', body });
    const data = await response.json();

    if (!response.ok) throw new Error(data.detail || 'Something went wrong.');
    showResult(data);
  } catch (error) {
    showError(error.message || 'Could not reach the server.');
  } finally {
    loading.hidden = true;
  }
}

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) analyse(fileInput.files[0]);
  fileInput.value = '';
});

['dragenter', 'dragover'].forEach((type) => dropzone.addEventListener(type, (event) => {
  event.preventDefault();
  dropzone.classList.add('dragging');
}));

['dragleave', 'drop'].forEach((type) => dropzone.addEventListener(type, (event) => {
  event.preventDefault();
  dropzone.classList.remove('dragging');
}));

dropzone.addEventListener('drop', (event) => {
  const file = event.dataTransfer.files[0];
  if (file) analyse(file);
});

opacity.addEventListener('input', setOpacity);
$('heatmap-toggle').addEventListener('change', setOpacity);

$('reset').addEventListener('click', () => {
  results.hidden = true;
  window.scrollTo({ top: 0, behavior: 'smooth' });
  fileInput.click();
});

fetch('/api/model')
  .then((response) => response.json())
  .then((info) => { $('param-count').textContent = info.parameters.toLocaleString(); })
  .catch(() => {});
