const canvas = document.getElementById("draw-canvas");
const ctx = canvas.getContext("2d");
const clearBtn = document.getElementById("clear-btn");
const predictBtn = document.getElementById("predict-btn");
const predictionEl = document.getElementById("prediction");
const top3El = document.getElementById("top3");

let isDrawing = false;

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
  return {
    x: evt.clientX - rect.left,
    y: evt.clientY - rect.top,
  };
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
  // Downsample 280x280 canvas to 28x28 by averaging 10x10 blocks.
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
          const gray = imageData[idx];
          sum += gray;
        }
      }
      out[row * 28 + col] = (sum / (block * block)) / 255.0;
    }
  }
  return out;
}

predictBtn.addEventListener("click", async () => {
  const pixels = canvasTo784();
  predictionEl.textContent = "Predicting...";
  top3El.innerHTML = "";

  const res = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pixels }),
  });
  const data = await res.json();

  if (!data.ok) {
    predictionEl.textContent = `Error: ${data.error}`;
    return;
  }

  predictionEl.textContent = `Predicted digit: ${data.predicted_digit}`;
  data.top3.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = `digit ${item.digit}: ${(item.prob * 100).toFixed(2)}%`;
    top3El.appendChild(li);
  });
});

resetCanvas();
