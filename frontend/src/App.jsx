import { useState, useEffect, useCallback } from "react";
import { API_BASE, fetchModels, predict, explain } from "./api";
import ImageUpload from "./components/ImageUpload";
import ModelSelector from "./components/ModelSelector";
import ResultsPanel from "./components/ResultsPanel";
import ComparePanel from "./components/ComparePanel";
import Header from "./components/Header";
import { Loader2, Scan, GitCompare } from "lucide-react";

function App() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [result, setResult] = useState(null);
  const [heatmap, setHeatmap] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showExplain, setShowExplain] = useState(false);
  const [error, setError] = useState(null);
  const [loadingModels, setLoadingModels] = useState(true);
  const [activeTab, setActiveTab] = useState("predict");

  useEffect(() => {
    let interval;

    const tryLoad = () => {
      fetchModels()
        .then((m) => {
          if (m.length > 0) {
            setModels(m);
            setSelectedModel((current) => current || m[0].model_name);
            setLoadingModels(false);
            setError(null);
            if (interval) clearInterval(interval);
          } else {
            setError("Backend warming up — downloading models from GCS. This takes ~1-2 minutes on first visit.");
          }
        })
        .catch((err) => {
          if (err.response?.status === 503) {
            setError("Backend warming up — downloading models from GCS. This takes ~1-2 minutes on first visit.");
          } else {
            setError(`Could not connect to API at ${API_BASE}.`);
            setLoadingModels(false);
            if (interval) clearInterval(interval);
          }
        });
    };

    tryLoad();
    interval = setInterval(tryLoad, 10000); // Poll every 10s

    return () => clearInterval(interval);
  }, []);

  const handleImageSelect = useCallback((file) => {
    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));
    setResult(null);
    setHeatmap(null);
    setError(null);
  }, []);

  const handlePredict = async () => {
    if (!imageFile || !selectedModel) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setHeatmap(null);
    setShowExplain(false);
    try {
      const res = await predict(imageFile, selectedModel);
      setResult(res);
    } catch (err) {
      setError(err.response?.data?.detail || "Prediction failed. Try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleExplain = async () => {
    if (!imageFile || !selectedModel) return;
    setShowExplain(true);
    setLoading(true);
    setError(null);
    try {
      const res = await explain(imageFile, selectedModel);
      setResult(res);
      setHeatmap(res.heatmap_base64);
    } catch (err) {
      setError(err.response?.data?.detail || "Explain failed. Try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setImageFile(null);
    setImagePreview(null);
    setResult(null);
    setHeatmap(null);
    setError(null);
    setShowExplain(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 via-white to-emerald-50">
      <Header />

      <main className="max-w-6xl mx-auto px-4 py-8">
        {/* Tab Navigation */}
        <div className="flex gap-2 mb-8">
          <button
            onClick={() => setActiveTab("predict")}
            className={`flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold text-sm transition-colors ${
              activeTab === "predict"
                ? "bg-emerald-600 text-white shadow-md"
                : "bg-white text-gray-600 border border-gray-200 hover:bg-gray-50"
            }`}
          >
            <Scan className="w-4 h-4" />
            Predict & Explain
          </button>
          <button
            onClick={() => setActiveTab("compare")}
            className={`flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold text-sm transition-colors ${
              activeTab === "compare"
                ? "bg-violet-600 text-white shadow-md"
                : "bg-white text-gray-600 border border-gray-200 hover:bg-gray-50"
            }`}
          >
            <GitCompare className="w-4 h-4" />
            Compare Models
          </button>
        </div>

        {/* Predict & Explain Tab */}
        {activeTab === "predict" && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Left Column — Upload + Controls */}
            <div className="space-y-6">
              <ImageUpload
                onImageSelect={handleImageSelect}
                imagePreview={imagePreview}
                onReset={handleReset}
              />

              {imageFile && (
                <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6 space-y-4">
                  <ModelSelector
                    models={models}
                    selected={selectedModel}
                    onChange={setSelectedModel}
                  />

                  <div className="flex gap-3">
                    <button
                      onClick={handlePredict}
                      disabled={loading}
                      className="flex-1 bg-emerald-600 hover:bg-emerald-700 disabled:bg-gray-300 text-white font-semibold py-3 px-6 rounded-xl transition-colors flex items-center justify-center gap-2"
                    >
                      {loading && !showExplain ? (
                        <Loader2 className="w-5 h-5 animate-spin" />
                      ) : null}
                      Predict
                    </button>

                    <button
                      onClick={handleExplain}
                      disabled={loading}
                      className="flex-1 bg-violet-600 hover:bg-violet-700 disabled:bg-gray-300 text-white font-semibold py-3 px-6 rounded-xl transition-colors flex items-center justify-center gap-2"
                    >
                      {loading && showExplain ? (
                        <Loader2 className="w-5 h-5 animate-spin" />
                      ) : null}
                      Explain (GradCAM)
                    </button>
                  </div>
                </div>
              )}

              {error && (
                <div className={`border px-4 py-4 rounded-xl text-sm flex items-start gap-3 ${
                  loadingModels
                    ? "bg-blue-50 border-blue-200 text-blue-700"
                    : "bg-red-50 border-red-200 text-red-700"
                }`}>
                  {loadingModels && <Loader2 className="w-5 h-5 animate-spin shrink-0 mt-0.5" />}
                  <span>{error}</span>
                </div>
              )}
            </div>

            {/* Right Column — Results */}
            <div>
              {result && (
                <ResultsPanel
                  result={result}
                  heatmap={heatmap}
                  imagePreview={imagePreview}
                />
              )}
            </div>
          </div>
        )}

        {/* Compare Tab */}
        {activeTab === "compare" && (
          <div className="max-w-4xl mx-auto space-y-6">
            <ImageUpload
              onImageSelect={handleImageSelect}
              imagePreview={imagePreview}
              onReset={handleReset}
            />

            {imageFile && (
              <ComparePanel
                models={models}
                imageFile={imageFile}
                imagePreview={imagePreview}
              />
            )}
          </div>
        )}
      </main>

      <footer className="text-center py-6 text-gray-400 text-sm">
        Crop Disease Detection &mdash; Powered by PyTorch &amp; FastAPI
      </footer>
    </div>
  );
}

export default App;
