import { useMemo, useState } from "react";
import "./App.css";

function pseudoRandom(seed) {
  const x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
}

function normalLikeNoise(seed) {
  let sum = 0;
  for (let i = 0; i < 12; i++) {
    sum += pseudoRandom(seed * 100 + i);
  }
  return sum - 6;
}

function predict(modelType, params, x) {
  if (modelType === "quadratic") {
    return params.a * x ** 2 + params.b * x + params.c;
  }
  return params.a * x + params.b;
}

function modelLabel(modelType) {
  return modelType === "quadratic" ? "二次関数" : "直線";
}

function formulaText(modelType, params, digits = 1) {
  const a = params.a.toFixed(digits);
  const bSign = params.b >= 0 ? "+" : "−";
  const b = Math.abs(params.b).toFixed(digits);

  if (modelType === "quadratic") {
    const cSign = params.c >= 0 ? "+" : "−";
    const c = Math.abs(params.c).toFixed(digits);
    return `y = ${a}x² ${bSign} ${b}x ${cSign} ${c}`;
  }

  return `y = ${a}x ${bSign} ${b}`;
}

function calculateRmse(data, modelType, params) {
  if (data.length === 0) return 0;
  const mse =
    data.reduce((sum, point) => {
      const yPred = predict(modelType, params, point.x);
      return sum + (point.y - yPred) ** 2;
    }, 0) / data.length;
  return Math.sqrt(mse);
}

function solveLinearSystem(matrix, vector) {
  const n = vector.length;
  const augmented = matrix.map((row, i) => [...row, vector[i]]);

  for (let col = 0; col < n; col++) {
    let pivotRow = col;

    for (let row = col + 1; row < n; row++) {
      if (Math.abs(augmented[row][col]) > Math.abs(augmented[pivotRow][col])) {
        pivotRow = row;
      }
    }

    [augmented[col], augmented[pivotRow]] = [
      augmented[pivotRow],
      augmented[col],
    ];

    const pivot = augmented[col][col];
    if (Math.abs(pivot) < 1e-12) return null;

    for (let j = col; j <= n; j++) {
      augmented[col][j] /= pivot;
    }

    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = augmented[row][col];

      for (let j = col; j <= n; j++) {
        augmented[row][j] -= factor * augmented[col][j];
      }
    }
  }

  return augmented.map((row) => row[n]);
}

function calculateLeastSquares(data, modelType) {
  if (data.length === 0) return null;

  if (modelType === "linear") {
    const n = data.length;
    const meanX = data.reduce((sum, point) => sum + point.x, 0) / n;
    const meanY = data.reduce((sum, point) => sum + point.y, 0) / n;

    const numerator = data.reduce(
      (sum, point) => sum + (point.x - meanX) * (point.y - meanY),
      0
    );

    const denominator = data.reduce(
      (sum, point) => sum + (point.x - meanX) ** 2,
      0
    );

    if (denominator === 0) return { a: 0, b: meanY };

    const a = numerator / denominator;
    const b = meanY - a * meanX;
    return { a, b };
  }

  const sums = data.reduce(
    (acc, point) => {
      const x = point.x;
      const y = point.y;

      acc.n += 1;
      acc.x += x;
      acc.x2 += x ** 2;
      acc.x3 += x ** 3;
      acc.x4 += x ** 4;
      acc.y += y;
      acc.xy += x * y;
      acc.x2y += x ** 2 * y;

      return acc;
    },
    { n: 0, x: 0, x2: 0, x3: 0, x4: 0, y: 0, xy: 0, x2y: 0 }
  );

  const matrix = [
    [sums.x4, sums.x3, sums.x2],
    [sums.x3, sums.x2, sums.x],
    [sums.x2, sums.x, sums.n],
  ];

  const vector = [sums.x2y, sums.xy, sums.y];
  const solution = solveLinearSystem(matrix, vector);

  if (!solution) return null;
  return { a: solution[0], b: solution[1], c: solution[2] };
}

function TogglePair({ label, value, onChange, leftLabel, rightLabel }) {
  return (
    <div className="toggle-block">
      <div className="small-label">{label}</div>
      <div className="toggle-pair">
        <button
          className={value === "linear" ? "toggle active" : "toggle"}
          onClick={() => onChange("linear")}
        >
          {leftLabel}
        </button>
        <button
          className={value === "quadratic" ? "toggle active" : "toggle"}
          onClick={() => onChange("quadratic")}
        >
          {rightLabel}
        </button>
      </div>
    </div>
  );
}

function CompactSlider({ label, value, min, max, step, onChange }) {
  const digits = step < 0.1 ? 2 : step < 1 ? 1 : 0;

  return (
    <div className="slider-block">
      <div className="slider-header">
        <label>{label}</label>
        <span>{value.toFixed(digits)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}

export default function App() {
  const [trueFunctionType, setTrueFunctionType] = useState("linear");
  const [manualModelType, setManualModelType] = useState("linear");
  const [leastSquaresModelType, setLeastSquaresModelType] = useState("linear");

  const [a, setA] = useState(1);
  const [b, setB] = useState(0);
  const [c, setC] = useState(0);
  const [sampleSize, setSampleSize] = useState(30);
  const [noise, setNoise] = useState(1.5);
  const [dataSeed, setDataSeed] = useState(1);
  const [leastSquaresModel, setLeastSquaresModel] = useState(null);

  const trueLinearParams = { a: 2, b: 1 };
  const trueQuadraticParams = { a: 0.25, b: -1.2, c: -2 };

  const width = 680;
  const height = 520;
  const padding = 44;
  const xMin = -10;
  const xMax = 10;
  const yMin = -35;
  const yMax = 35;

  const xToSvg = (x) =>
    padding + ((x - xMin) / (xMax - xMin)) * (width - padding * 2);

  const yToSvg = (y) =>
    height - padding - ((y - yMin) / (yMax - yMin)) * (height - padding * 2);

  const makeModelPoints = (modelType, params) => {
    const result = [];

    for (let x = xMin; x <= xMax; x += 0.1) {
      const y = predict(modelType, params, x);
      result.push(`${xToSvg(x)},${yToSvg(y)}`);
    }

    return result.join(" ");
  };

  const manualParams = useMemo(() => {
    return manualModelType === "quadratic" ? { a, b, c } : { a, b };
  }, [manualModelType, a, b, c]);

  const modelPoints = useMemo(() => {
    return makeModelPoints(manualModelType, manualParams);
  }, [manualModelType, manualParams]);

  const leastSquaresPoints = useMemo(() => {
    if (!leastSquaresModel) return "";
    return makeModelPoints(leastSquaresModel.type, leastSquaresModel.params);
  }, [leastSquaresModel]);

  const data = useMemo(() => {
    const result = [];
    const trueParams =
      trueFunctionType === "quadratic" ? trueQuadraticParams : trueLinearParams;

    for (let i = 0; i < sampleSize; i++) {
      const r = pseudoRandom(dataSeed * 1000 + i);
      const x = xMin + r * (xMax - xMin);
      const yTrue = predict(trueFunctionType, trueParams, x);
      const y = yTrue + noise * normalLikeNoise(dataSeed * 2000 + i);

      result.push({ x, y, yTrue });
    }

    return result;
  }, [sampleSize, noise, dataSeed, trueFunctionType]);

  const manualRmse = useMemo(() => {
    return calculateRmse(data, manualModelType, manualParams);
  }, [data, manualModelType, manualParams]);

  const leastSquaresRmse = useMemo(() => {
    if (!leastSquaresModel) return null;
    return calculateRmse(data, leastSquaresModel.type, leastSquaresModel.params);
  }, [data, leastSquaresModel]);

  const gridLines = [];

  for (let i = -10; i <= 10; i += 1) {
    gridLines.push(
      <line
        key={`grid-x-${i}`}
        x1={xToSvg(i)}
        y1={yToSvg(yMin)}
        x2={xToSvg(i)}
        y2={yToSvg(yMax)}
        className="grid-line"
      />
    );
  }

  for (let i = -30; i <= 30; i += 5) {
    gridLines.push(
      <line
        key={`grid-y-${i}`}
        x1={xToSvg(xMin)}
        y1={yToSvg(i)}
        x2={xToSvg(xMax)}
        y2={yToSvg(i)}
        className="grid-line"
      />
    );
  }

  const tickLabels = [];

  for (let i = -10; i <= 10; i += 5) {
    if (i !== 0) {
      tickLabels.push(
        <text
          key={`x-label-${i}`}
          x={xToSvg(i)}
          y={yToSvg(0) + 18}
          textAnchor="middle"
          className="tick-label"
        >
          {i}
        </text>
      );
    }
  }

  for (let i = -30; i <= 30; i += 10) {
    if (i !== 0) {
      tickLabels.push(
        <text
          key={`y-label-${i}`}
          x={xToSvg(0) - 12}
          y={yToSvg(i) + 4}
          textAnchor="end"
          className="tick-label"
        >
          {i}
        </text>
      );
    }
  }

  const reset = () => {
    setTrueFunctionType("linear");
    setManualModelType("linear");
    setLeastSquaresModelType("linear");
    setA(1);
    setB(0);
    setC(0);
    setSampleSize(30);
    setNoise(1.5);
    setDataSeed(1);
    setLeastSquaresModel(null);
  };

  const regenerateData = () => {
    setDataSeed((prev) => prev + 1);
    setLeastSquaresModel(null);
  };

  const runLeastSquares = () => {
    const result = calculateLeastSquares(data, leastSquaresModelType);
    if (!result) return;
    setLeastSquaresModel({ type: leastSquaresModelType, params: result });
  };

  const changeTrueFunctionType = (type) => {
    setTrueFunctionType(type);
    setLeastSquaresModel(null);
  };

  const changeLeastSquaresModelType = (type) => {
    setLeastSquaresModelType(type);
    setLeastSquaresModel(null);
  };

  return (
    <div className="page">
      <div className="app-container">
        <header className="header">
          <div>
            <h1>関数パラメータ・シミュレータ</h1>
            <p>
              データ、手動モデル、最小二乗法モデルを切り替えて比較します。
            </p>
          </div>

          <div className="legend">
            <div>
              <span className="legend-line manual"></span>手動
            </div>
            <div>
              <span className="legend-line least"></span>最小二乗法
            </div>
            <div>
              <span className="legend-dot"></span>データ
            </div>
          </div>
        </header>

        <main className="main-layout">
          <section className="plot-card">
            <svg
              viewBox={`0 0 ${width} ${height}`}
              className="plot"
              aria-label="function plot"
            >
              {gridLines}

              <line
                x1={xToSvg(xMin)}
                y1={yToSvg(0)}
                x2={xToSvg(xMax)}
                y2={yToSvg(0)}
                className="axis-line"
              />

              <line
                x1={xToSvg(0)}
                y1={yToSvg(yMin)}
                x2={xToSvg(0)}
                y2={yToSvg(yMax)}
                className="axis-line"
              />

              {tickLabels}

              <text
                x={xToSvg(xMax) + 18}
                y={yToSvg(0) + 5}
                className="axis-label"
              >
                x
              </text>

              <text
                x={xToSvg(0) - 5}
                y={yToSvg(yMax) - 16}
                className="axis-label"
              >
                y
              </text>

              {data.map((point, index) => (
                <circle
                  key={`data-${index}`}
                  cx={xToSvg(point.x)}
                  cy={yToSvg(point.y)}
                  r="4.5"
                  className="data-point"
                />
              ))}

              {leastSquaresModel && (
                <polyline
                  points={leastSquaresPoints}
                  fill="none"
                  className="least-line"
                />
              )}

              <polyline
                points={modelPoints}
                fill="none"
                className="manual-line"
              />
            </svg>
          </section>

          <section className="side-panel">
            <div className="control-card">
              <div className="toggle-row">
                <TogglePair
                  label="データ"
                  value={trueFunctionType}
                  onChange={changeTrueFunctionType}
                  leftLabel="データ1"
                  rightLabel="データ2"
                />

                <TogglePair
                  label="手動モデル"
                  value={manualModelType}
                  onChange={setManualModelType}
                  leftLabel="直線"
                  rightLabel="二次"
                />

                <TogglePair
                  label="最小二乗法"
                  value={leastSquaresModelType}
                  onChange={changeLeastSquaresModelType}
                  leftLabel="直線"
                  rightLabel="二次"
                />
              </div>

              <div className="slider-grid">
                <CompactSlider
                  label={manualModelType === "quadratic" ? "a：x²係数" : "a：傾き"}
                  value={a}
                  min={-5}
                  max={5}
                  step={0.05}
                  onChange={setA}
                />

                <CompactSlider
                  label={manualModelType === "quadratic" ? "b：x係数" : "b：切片"}
                  value={b}
                  min={-10}
                  max={10}
                  step={0.05}
                  onChange={setB}
                />

                {manualModelType === "quadratic" && (
                  <CompactSlider
                    label="c：切片"
                    value={c}
                    min={-20}
                    max={20}
                    step={0.1}
                    onChange={setC}
                  />
                )}

                <CompactSlider
                  label="サンプルサイズ"
                  value={sampleSize}
                  min={5}
                  max={200}
                  step={5}
                  onChange={(v) => {
                    setSampleSize(v);
                    setLeastSquaresModel(null);
                  }}
                />

                <CompactSlider
                  label="ノイズ"
                  value={noise}
                  min={0}
                  max={6}
                  step={0.1}
                  onChange={(v) => {
                    setNoise(v);
                    setLeastSquaresModel(null);
                  }}
                />
              </div>

              <div className="button-row">
                <button className="primary-button" onClick={runLeastSquares}>
                  最小二乗法
                </button>

                <button className="secondary-button" onClick={regenerateData}>
                  再生成
                </button>

                <button className="secondary-button" onClick={reset}>
                  初期値
                </button>
              </div>
            </div>

            <div className="result-card">
              <div className="formula-box">
                {formulaText(manualModelType, manualParams, 2)}
              </div>

              <div className="rmse-grid">
                <div className="rmse-box manual-rmse">
                  <div className="rmse-label">手動RMSE</div>
                  <div className="rmse-value">{manualRmse.toFixed(3)}</div>
                </div>

                <div className="rmse-box least-rmse">
                  <div className="rmse-label">最小二乗RMSE</div>
                  <div className="rmse-value">
                    {leastSquaresRmse === null
                      ? "—"
                      : leastSquaresRmse.toFixed(3)}
                  </div>
                </div>
              </div>

              {leastSquaresModel ? (
                <div className="least-result">
                  <div className="small-label">
                    最小二乗法モデル：{modelLabel(leastSquaresModel.type)}
                  </div>
                  <div className="least-formula">
                    {formulaText(
                      leastSquaresModel.type,
                      leastSquaresModel.params,
                      3
                    )}
                  </div>
                </div>
              ) : (
                <div className="hint-box">
                  「最小二乗法」ボタンを押すと、選択したモデルでデータに最もよく合う関数を表示します。
                </div>
              )}

              <div className="note-box">
                データ生成の関数は非表示です。現在は{" "}
                <strong>{trueFunctionType === "linear" ? "データ1" : "データ2"}</strong>{" "}
                を表示しています。RMSEは予測値とデータ点のズレを表し、小さいほど当てはまりがよいことを意味します。
              </div>
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}