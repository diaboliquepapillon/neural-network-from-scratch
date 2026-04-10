const canvas = document.getElementById("draw-canvas");
const ctx = canvas.getContext("2d");
const clearBtn = document.getElementById("clear-btn");
const predictBtn = document.getElementById("predict-btn");
const predictionEl = document.getElementById("prediction");
const top3El = document.getElementById("top3");

let isDrawing = false;
let params = null;

function resetCanvas() {
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function drawAt(x, y) {
  ctx.fillStyle = "white";
  ctx.beginPath();
  ctx.arc(x, y, 12, 0, Math.PI * 2);
  ctx.fill();
}

function pointerPos(evt) {
  const rect = canvas.getBoundingClientRect();
  return { x: evt.clientX - rect.left, y: evt.clientY - rect.top };
}

canvas.addEventListener("pointerdown", (evt) => {
  isDrawing = true;
  const p = pointerPos(evt);
  drawAt(p.x, p.y);
});

canvas.addEventListener("pointermove", (evt) => {
  if (!isDrawing) return;
  const p = pointerPos(evt);
  drawAt(p.x, p.y);
});

window.addEventListener("pointerup", () => {
  isDrawing = false;
});

clearBtn.addEventListener("click", () => {
  resetCanvas();
  predictionEl.textContent = "Draw something and click Predict.";
  top3El.innerHTML = "";
});

function canvasTo784() {
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  const out = new Array(784).fill(0);
  const block = 10;
  for (let row = 0; row < 28; row++) {
    for (let col = 0; col < 28; col++) {
      let sum = 0;
      for (let dy = 0; dy < block; dy++) {
        for (let dx = 0; dx < block; dx++) {
          const x = col * block + dx;
          const y = row * block + dy;
          const idx = (y * canvas.width + x) * 4;
          sum += imageData[idx];
        }
      }
      out[row * 28 + col] = (sum / (block * block)) / 255.0;
    }
  }
  return out;
}

function dense(x, W, b) {
  const outDim = b[0].length;
  const inDim = x.length;
  const y = new Array(outDim).fill(0);
  for (let j = 0; j < outDim; j++) {
    let s = b[0][j];
    for (let i = 0; i < inDim; i++) s += x[i] * W[i][j];
    y[j] = s;
  }
  return y;
}

function relu(v) {
  return v.map((x) => (x > 0 ? x : 0));
}

function softmax(logits) {
  const maxV = Math.max(...logits);
  const exps = logits.map((x) => Math.exp(x - maxV));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / s);
}

function predictFromPixels(x) {
  const z1 = dense(x, params.W1, params.b1);
  const a1 = relu(z1);
  const z2 = dense(a1, params.W2, params.b2);
  const a2 = relu(z2);
  const z3 = dense(a2, params.W3, params.b3);
  return softmax(z3);
}

predictBtn.addEventListener("click", () => {
  if (!params) return;
  const x = canvasTo784();
  const probs = predictFromPixels(x);
  const pred = probs.indexOf(Math.max(...probs));

  predictionEl.textContent = `Predicted digit: ${pred}`;
  top3El.innerHTML = "";
  const sorted = probs
    .map((p, i) => ({ digit: i, prob: p }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, 3);
  sorted.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = `digit ${item.digit}: ${(item.prob * 100).toFixed(2)}%`;
    top3El.appendChild(li);
  });
});

async function loadWeights() {
  try {
    const res = await fetch("./weights.json");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    params = await res.json();
    predictionEl.textContent = "Draw something and click Predict.";
  } catch (err) {
    predictionEl.textContent = `Failed to load weights.json: ${err.message}`;
    predictBtn.disabled = true;
  }
}

resetCanvas();
loadWeights();
