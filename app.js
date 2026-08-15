let ing2id = null;
let id2ing = null;
let session = null;
const currentIngredients = [];

const statusEl = document.getElementById('status');
const ingInput = document.getElementById('ingInput');
const addBtn = document.getElementById('addBtn');
const predictBtn = document.getElementById('predictBtn');
const chipsContainer = document.getElementById('ingredientChips');
const resultContainer = document.getElementById('resultContainer');
const resultList = document.getElementById('resultList');

// 辞書とONNXモデルの初期化
async function init() {
  try {
    const [ingData, idData] = await Promise.all([
      fetch('./ing2id.json').then(res => res.json()),
      fetch('./id2ing.json').then(res => res.json())
    ]);

    ing2id = ingData;
    id2ing = idData;

    session = await ort.InferenceSession.create('./recipe_encoder.onnx');

    statusEl.textContent = '✅ モデルの準備が完了しました';
    statusEl.style.color = '#059669';
    ingInput.disabled = false;
    addBtn.disabled = false;
  } catch (err) {
    statusEl.textContent = '❌ 初期化エラー: ' + err.message;
    statusEl.style.color = '#dc2626';
    console.error(err);
  }
}

// 食材バッジ描画
function renderChips() {
  chipsContainer.innerHTML = '';
  currentIngredients.forEach(ing => {
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.textContent = ing;
    chipsContainer.appendChild(chip);
  });
  predictBtn.disabled = currentIngredients.length === 0;
}

// 食材追加処理
function addIngredient() {
  const val = ingInput.value.trim().toLowerCase();
  if (!val) return;
  currentIngredients.push(val);
  ingInput.value = '';
  renderChips();
}

addBtn.addEventListener('click', addIngredient);
ingInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') addIngredient();
});

// 推論処理
predictBtn.addEventListener('click', async () => {
  if (!session || currentIngredients.length === 0) return;

  predictBtn.disabled = true;
  predictBtn.textContent = '推論中...';

  try {
    const padId = ing2id["[PAD]"] ?? 0;
    const maskId = ing2id["[MASK]"] ?? 1;
    const unkId = ing2id["[UNK]"] ?? 2;

    let inputIds = currentIngredients.map(ing => ing2id[ing] ?? unkId);
    inputIds.push(maskId);

    while (inputIds.length < 16) {
      inputIds.push(padId);
    }
    inputIds = inputIds.slice(0, 16);

    const tensor = new ort.Tensor('int64', BigInt64Array.from(inputIds.map(BigInt)), [1, 16]);
    const results = await session.run({ input: tensor });
    const logits = results.output.data;

    const vocabSize = Object.keys(ing2id).length;
    const pooledLogits = new Float32Array(vocabSize);

    // Mean Pooling
    for (let seq = 0; seq < 16; seq++) {
      for (let v = 0; v < vocabSize; v++) {
        pooledLogits[v] += logits[seq * vocabSize + v];
      }
    }
    for (let v = 0; v < vocabSize; v++) {
      pooledLogits[v] /= 16;
    }

    const scored = [];
    for (let v = 0; v < vocabSize; v++) {
      scored.push({ id: v, score: pooledLogits[v] });
    }
    scored.sort((a, b) => b.score - a.score);

    const top5 = scored.slice(0, 5);

    resultList.innerHTML = '';
    top5.forEach(item => {
      const row = document.createElement('div');
      row.className = 'result-item';
      row.innerHTML = `
        <span>${id2ing[item.id.toString()] || '[不明]'}</span>
        <span class="score">${item.score.toFixed(2)}</span>
      `;
      resultList.appendChild(row);
    });

    resultContainer.style.display = 'block';
  } catch (err) {
    alert('推論に失敗しました: ' + err.message);
    console.error(err);
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = '予測する';
  }
});

init();
