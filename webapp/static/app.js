const MAX_BYTES = 10 * 1024 * 1024;
const VIEWER_HEIGHT = 480;
// The API keys are "Tumor" / "No tumor"; the page displays British spelling
const DISPLAY = { Tumor: 'Tumour', 'No tumor': 'No tumour' };
const MEANING = {
  Tumor: 'The model sees a tumour pattern. This is not a diagnosis; a radiologist has to review it.',
  'No tumor': 'The model found no tumour pattern. It does miss some tumours, so this does not rule one out.',
};

const $ = (id) => document.getElementById(id);
const dropzone = $('dropzone');
const fileInput = $('file-input');
const errorBox = $('error');
const loading = $('loading');
const results = $('results');
const viewer = $('viewer');
const frame = $('viewer-frame');
const heatmap = $('heatmap');
const divider = $('divider');
const compare = $('compare');
const opacity = $('opacity');

const percent = (value) => `${(value * 100).toFixed(1)}%`;
const signed = (value) => `${value >= 0 ? '+' : '−'}${Math.abs(value * 100).toFixed(1)} pts`;
// Overwritten by /api/model; defaults match the server so the page works if that call fails
let limits = { max_background_heat: 0.55, max_score_change: 0.2, min_heatmap_agreement: 0.35 };
// Fractions of the padded square image where the real scan lies
let box = { x: 0, y: 0, w: 1, h: 1 };

const STATUS_TONE = { consistent: 'ok', sensitive: 'warn', poor_quality: 'warn', review: 'review' };

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function showQuality(quality) {
  const warned = quality.checks.filter((c) => c.status !== 'ok').length;
  $('quality-summary').textContent = warned
    ? `${warned} of ${quality.checks.length} checks need attention`
    : `All ${quality.checks.length} checks passed`;
  $('quality-summary').dataset.tone = warned ? 'warn' : 'ok';
  $('quality-details').open = quality.checks.some((c) => c.status === 'warning');
  $('quality-rows').replaceChildren(...quality.checks.map((check) => {
    const row = el('tr');
    row.dataset.status = check.status;
    const name = el('th', '', check.label);
    name.scope = 'row';
    if (check.message) name.append(el('span', 'check-message', check.message));
    const state = el('td', 'state');
    state.append(el('span', 'pill', check.status === 'ok' ? 'OK' : 'Warning'));
    row.append(name, el('td', 'num', check.value), el('td', 'num muted', check.expected), state);
    return row;
  }));
}

function reliabilityWarnings(rel) {
  // Low confidence is already called out in the verdict card
  const list = [];
  const f = rel.followup;
  if (rel.background_heat > limits.max_background_heat) {
    list.push(`${Math.round(rel.background_heat * 100)}% of the heatmap weight is outside the head, so the model may be using the wrong cues.`);
  }
  if (f && f.label_changed) list.push('The follow-up test changed the predicted label.');
  if (f && Math.abs(f.score_change) > limits.max_score_change) list.push('The tumour score moved a lot under a small, controlled change.');
  if (f && f.predicted_tumor && f.heatmap_agreement != null && f.heatmap_agreement < limits.min_heatmap_agreement) list.push('The heatmap moved to a different region in the follow-up test.');
  if (rel.status === 'poor_quality') list.push('Image quality is too low for the result to mean much.');
  if (rel.human_review) list.push('Flagged for human review in this demo.');
  return list;
}

function showReliability(data) {
  const rel = data.reliability;
  const badge = $('status-badge');
  badge.textContent = rel.status_label;
  badge.dataset.tone = STATUS_TONE[rel.status] || 'warn';

  const f = rel.followup;
  $('followup').hidden = !f;
  if (f) {
    $('followup-action').textContent = `Follow-up test: ${f.action}`;
    $('score-initial').textContent = percent(data.probabilities.Tumor);
    $('score-followup').textContent = percent(f.tumor_probability);
    $('score-change').textContent = signed(f.score_change);
    $('score-change').dataset.tone = Math.abs(f.score_change) > limits.max_score_change ? 'warn' : 'ok';
    $('followup-tool').textContent = `${f.tool}. Tumour score shown; label ${f.label_changed ? 'changed' : 'unchanged'}${f.heatmap_agreement != null ? `; heatmap agreement ${f.heatmap_agreement.toFixed(2)}` : ''}.`;
  }

  // Quality messages already sit in the quality table; point to them instead of repeating
  const n = data.warnings.length;
  const warnings = [...(n ? [`${n} image quality warning${n > 1 ? 's' : ''}, see Image quality.`] : []), ...reliabilityWarnings(rel)];
  $('reliability-warnings').replaceChildren(...warnings.map((text) => el('li', '', text)));

  $('trace').replaceChildren(...rel.trace.map((item) => {
    const li = el('li');
    li.append(el('span', 'trace-step', item.step), el('span', 'trace-detail', item.detail));
    return li;
  }));
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function setOpacity() {
  const visible = $('heatmap-toggle').checked;
  heatmap.style.opacity = visible ? opacity.value / 100 : 0;
  $('opacity-value').textContent = `${opacity.value}%`;
  opacity.disabled = !visible;
  viewer.classList.toggle('no-heatmap', !visible);
}

function setCompare() {
  // Heatmap shows left of the divider. The heatmap spans the whole padded frame, so map the
  // divider (a fraction of the visible box) into frame coordinates for the clip
  const hidden = 100 - compare.value;
  const right = 100 - (box.x + box.w * (compare.value / 100)) * 100;
  heatmap.style.clipPath = `inset(0 ${right}% 0 0)`;
  divider.style.transform = `translateX(-${hidden}%)`;
}

function probabilityRow(name, p, win) {
  const row = document.createElement('div');
  row.className = 'prob-row';
  if (win) row.dataset.win = '';
  const bar = el('div', 'bar');
  bar.title = 'decision threshold 50%';
  bar.setAttribute('role', 'img');
  bar.setAttribute('aria-label', `${percent(p)}, decision threshold 50%`);
  bar.append(el('div', 'bar-fill'));
  row.append(el('dt', '', DISPLAY[name] || name), bar, el('dd', '', percent(p)));
  return row;
}

function sizeViewer() {
  // The viewer has the scan's own shape (content_box of a square), at most VIEWER_HEIGHT tall;
  // the frame is the full padded square, scaled and shifted so only the content shows
  const ratio = box.w / box.h;
  viewer.style.aspectRatio = `${ratio}`;
  viewer.style.width = `min(100%, ${Math.round(VIEWER_HEIGHT * ratio)}px)`;
  frame.style.width = `${100 / box.w}%`;
  frame.style.height = `${100 / box.h}%`;
  frame.style.left = `${(-box.x / box.w) * 100}%`;
  frame.style.top = `${(-box.y / box.h) * 100}%`;
}

function showResult(data) {
  const b = data.content_box;
  box = b && b.w > 0 && b.h > 0 ? b : { x: 0, y: 0, w: 1, h: 1 };
  sizeViewer();
  $('scan').src = data.image;
  heatmap.src = data.heatmap;
  $('model-view').src = data.model_view;
  $('original').src = data.original;
  $('cropped').src = data.image;

  const prediction = $('prediction');
  prediction.textContent = DISPLAY[data.prediction] || data.prediction;
  prediction.className = `prediction ${data.prediction === 'Tumor' ? 'tumor' : 'clear'}`;
  $('meaning').textContent = MEANING[data.prediction] || '';

  const entries = Object.entries(data.probabilities);
  const top = Math.max(...entries.map(([, p]) => p));
  $('probabilities').replaceChildren(...entries.map(([name, p]) => probabilityRow(name, p, p === top)));

  // A result that is not "consistent" carries its trust status in the verdict itself
  const rel = data.reliability;
  const flagged = rel.status !== 'consistent' || rel.human_review;
  $('verdict').classList.toggle('flagged', flagged);
  const uncertain = $('uncertain');
  uncertain.hidden = !flagged && !rel.low_confidence;
  uncertain.textContent = rel.low_confidence
    ? `The score is close to the decision threshold, so the model is unsure about this scan.${flagged ? ' Flagged for human review.' : ''}`
    : `Flagged for human review: ${rel.status_label}.`;
  showQuality(data.quality);
  showReliability(data);

  compare.value = 100;
  setCompare();
  setOpacity();
  results.hidden = false;

  // Fill the bars on the next frame so they animate from empty
  const fills = results.querySelectorAll('.bar-fill');
  requestAnimationFrame(() => requestAnimationFrame(() => {
    fills.forEach((fill, i) => { fill.style.transform = `scaleX(${entries[i][1]})`; });
  }));
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

document.querySelectorAll('.sample').forEach((button) => button.addEventListener('click', async () => {
  try {
    const response = await fetch(button.dataset.src);
    if (!response.ok) throw new Error();
    const blob = await response.blob();
    analyse(new File([blob], button.dataset.src.split('/').pop(), { type: blob.type || 'image/jpeg' }));
  } catch {
    showError('Could not load that sample.');
  }
}));

compare.addEventListener('input', setCompare);
opacity.addEventListener('input', setOpacity);
$('heatmap-toggle').addEventListener('change', setOpacity);

$('reset').addEventListener('click', () => {
  results.hidden = true;
  window.scrollTo({ top: 0, behavior: 'smooth' });
  fileInput.click();
});

// Decision trace: open on desktop, folded on phones to keep the verdict column short
$('trace-details').open = matchMedia('(min-width: 861px)').matches;

// Scroll-edge effect: the header only shows its fade once content is underneath it
const header = $('site-header');
const onScroll = () => header.classList.toggle('scrolled', window.scrollY > 4);
window.addEventListener('scroll', onScroll, { passive: true });
onScroll();

fetch('/api/model')
  .then((response) => response.json())
  .then((info) => {
    limits = { ...limits, ...info.reliability_thresholds };
    const size = `${info.input_size}×${info.input_size}`;
    $('param-count').textContent = info.parameters.toLocaleString();
    document.querySelectorAll('.input-size').forEach((el) => { el.textContent = size; });
    $('model-chips').replaceChildren(...[
      `${info.parameters.toLocaleString()} parameters`, `${size} input`, 'Trained from scratch', 'PyTorch',
    ].map((text) => el('li', '', text)));
  })
  .catch(() => {});
