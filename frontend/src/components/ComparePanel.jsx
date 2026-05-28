import { useEffect, useMemo, useState } from "react";
import { explain } from "../api";
import { Loader2, GitCompare } from "lucide-react";

export default function ComparePanel({ models, imageFile, imagePreview }) {
  const [selectedModels, setSelectedModels] = useState(["", "", ""]);
  const [results, setResults] = useState([null, null, null]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const modelOptions = useMemo(() => models.map((m) => m.model_name), [models]);

  useEffect(() => {
    if (modelOptions.length === 0) return;
    setSelectedModels((current) => {
      if (current.some(Boolean)) return current;
      return [
        modelOptions[0] || "",
        modelOptions[1] || "",
        modelOptions[2] || "",
      ];
    });
  }, [modelOptions]);

  const handleModelChange = (idx, value) => {
    const updated = [...selectedModels];
    updated[idx] = value;
    setSelectedModels(updated);
  };

  const handleCompare = async () => {
    const modelsToCompare = selectedModels.filter((m) => m !== "");
    if (modelsToCompare.length < 2) {
      setError("Select at least 2 models to compare.");
      return;
    }
    if (!imageFile) {
      setError("Upload an image first.");
      return;
    }
    if (new Set(modelsToCompare).size !== modelsToCompare.length) {
      setError("Choose different models for each comparison slot.");
      return;
    }

    setLoading(true);
    setError(null);
    setResults([null, null, null]);

    try {
      const promises = selectedModels.map((modelName) => {
        if (!modelName) return Promise.resolve(null);
        return explain(imageFile, modelName);
      });
      const res = await Promise.all(promises);
      setResults(res);
    } catch (err) {
      setError(err.response?.data?.detail || "Comparison failed. Try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Model Selectors */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6 space-y-4">
        <div className="flex items-center gap-2 mb-2">
          <GitCompare className="w-5 h-5 text-violet-600" />
          <h3 className="text-lg font-bold text-gray-900">
            Compare GradCAM Across Models
          </h3>
        </div>
        <p className="text-sm text-gray-500">
          Select 2–3 models to see where each focuses its attention
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {[0, 1, 2].map((idx) => (
            <div key={idx}>
              <label className="text-xs font-semibold text-gray-500 mb-1 block">
                Model {idx + 1} {idx < 2 && <span className="text-red-400">*</span>}
              </label>
              <select
                value={selectedModels[idx]}
                onChange={(e) => handleModelChange(idx, e.target.value)}
                className="w-full bg-gray-50 border border-gray-200 rounded-xl px-3 py-2.5 text-sm text-gray-800 font-medium focus:outline-none focus:ring-2 focus:ring-violet-400 focus:border-transparent"
              >
                <option value="">
                  {idx === 2 ? "(Optional)" : "Select model..."}
                </option>
                {models.map((m) => (
                  <option
                    key={m.model_name}
                    value={m.model_name}
                    disabled={selectedModels.includes(m.model_name) && selectedModels[idx] !== m.model_name}
                  >
                    {m.display_name}
                  </option>
                ))}
              </select>
            </div>
          ))}
        </div>

        <button
          onClick={handleCompare}
          disabled={loading || !imageFile}
          className="w-full bg-violet-600 hover:bg-violet-700 disabled:bg-gray-300 text-white font-semibold py-3 px-6 rounded-xl transition-colors flex items-center justify-center gap-2"
        >
          {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : null}
          {loading ? "Generating heatmaps..." : "Compare GradCAM"}
        </button>

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-xl text-sm">
            {error}
          </div>
        )}
      </div>

      {/* Comparison Results */}
      {results.some((r) => r !== null) && (
        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
          <h3 className="text-sm font-semibold text-gray-700 mb-4">
            GradCAM Comparison
          </h3>

          {/* Original image */}
          <div className="mb-4">
            <p className="text-xs text-gray-500 mb-2 text-center">
              Original Image
            </p>
            <img
              src={imagePreview}
              alt="Original"
              className="mx-auto h-40 rounded-xl object-contain bg-gray-50"
            />
          </div>

          {/* Heatmaps grid */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {results.map((res, idx) => {
              if (!res) return null;
              const modelInfo = models.find(
                (m) => m.model_name === selectedModels[idx]
              );
              return (
                <div key={idx} className="text-center">
                  <p className="text-xs font-semibold text-gray-700 mb-1">
                    {modelInfo?.display_name || selectedModels[idx]}
                  </p>
                  <p className="text-xs text-gray-400 mb-2">
                    {res.prediction?.replace(/_/g, " ")} (
                    {(res.confidence * 100).toFixed(1)}%)
                  </p>
                  {res.heatmap_base64 ? (
                    <img
                      src={`data:image/png;base64,${res.heatmap_base64}`}
                      alt={`GradCAM ${selectedModels[idx]}`}
                      className="w-full rounded-xl object-contain bg-gray-50 h-48"
                    />
                  ) : (
                    <div className="h-48 bg-gray-100 rounded-xl flex items-center justify-center text-gray-400 text-sm">
                      No heatmap
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Insight note */}
          <div className="mt-4 bg-violet-50 border border-violet-100 rounded-xl px-4 py-3">
            <p className="text-xs text-violet-700">
              <strong>How to read:</strong> Red/yellow areas = high attention
              (where the model looks). Compare focus regions to understand which
              model best identifies disease-affected tissue vs background.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
