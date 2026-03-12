import { useEffect, useState } from "react";

function createId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function formatConfidence(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  return `${(value * 100).toFixed(2)}%`;
}

function getPredictionStatus(prediction) {
  const normalizedPrediction = (prediction || "").toLowerCase();
  if (normalizedPrediction === "notumor") {
    return { label: "normal", className: "status-normal" };
  }
  return { label: "tumor detected", className: "status-tumor-detected" };
}

export default function App() {
  const [rows, setRows] = useState([]);

  useEffect(() => {
    return () => {
      rows.forEach((row) => {
        if (row.previewUrl) URL.revokeObjectURL(row.previewUrl);
      });
    };
  }, [rows]);

  const updateRow = (rowId, patch) => {
    setRows((current) =>
      current.map((row) => (row.id === rowId ? { ...row, ...patch } : row))
    );
  };

  const predictImage = async (rowId, file) => {
    updateRow(rowId, { status: "processing", error: "" });

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("/predict", {
        method: "POST",
        body: formData
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed.");
      }

      updateRow(rowId, {
        status: "done",
        prediction: data.prediction || "-",
        confidence: data.confidence,
        explanation: data.explanation || "-"
      });
    } catch (error) {
      updateRow(rowId, {
        status: "error",
        error: error.message || "Prediction request failed."
      });
    }
  };

  const onFilesSelected = (event) => {
    const files = Array.from(event.target.files || []);
    if (!files.length) return;

    const newRows = files.map((file) => ({
      id: createId(),
      fileName: file.name,
      previewUrl: URL.createObjectURL(file),
      status: "queued",
      prediction: "-",
      confidence: null,
      explanation: "Waiting for prediction...",
      error: ""
    }));

    setRows((current) => [...current, ...newRows]);
    newRows.forEach((row, index) => predictImage(row.id, files[index]));
    event.target.value = "";
  };

  const clearAll = () => {
    rows.forEach((row) => {
      if (row.previewUrl) URL.revokeObjectURL(row.previewUrl);
    });
    setRows([]);
  };

  return (
    <div className="app">
      <header className="hero">
        <h1>Brain Tumor Classification</h1>
        <p>Generating Human-Readable MRI Explanations Using Deep Learning</p>
      </header>

      <section className="uploader">
        <label className="upload-label" htmlFor="image-upload">
          Select image files
        </label>
        <input
          id="image-upload"
          type="file"
          accept="image/*"
          multiple
          onChange={onFilesSelected}
        />
        <button className="clear-btn" type="button" onClick={clearAll} disabled={!rows.length}>
          Clear Rows
        </button>
      </section>

      <section className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Image</th>
              <th>File</th>
              <th>Status</th>
              <th>Prediction</th>
              <th>Confidence</th>
              <th>Human Explanation</th>
            </tr>
          </thead>
          <tbody>
            {!rows.length ? (
              <tr>
                <td colSpan={6} className="empty-state">
                  No images uploaded yet.
                </td>
              </tr>
            ) : (
              rows.map((row) => {
                const doneStatus =
                  row.status === "done" ? getPredictionStatus(row.prediction) : null;
                const statusClassName = doneStatus
                  ? doneStatus.className
                  : `status-${row.status}`;
                const statusLabel = doneStatus ? doneStatus.label : row.status;

                return (
                  <tr key={row.id}>
                    <td>
                      <img className="thumb" src={row.previewUrl} alt={row.fileName} />
                    </td>
                    <td>{row.fileName}</td>
                    <td>
                      <span className={`status ${statusClassName}`}>{statusLabel}</span>
                    </td>
                    <td>{row.prediction}</td>
                    <td>{formatConfidence(row.confidence)}</td>
                    <td className="explanation">
                      {row.status === "error" ? row.error : row.explanation}
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </section>
    </div>
  );
}
