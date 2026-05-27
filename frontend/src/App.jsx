import { useState, useEffect, useCallback } from "react";
import { fetchModels, predict, explain } from "./api";
import ImageUpload from "./components/ImageUpload";
import ModelSelector from "./components/ModelSelector";
import ResultsPanel from "./components/ResultsPanel";
import Header from "./components/Header";
import { Loader2 } from "lucide-react";

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

  useEffect(() => {
    fetchModels()
      .then((m) => {
        setModels(m);
        if (m.length > 0) setSelectedModel(m[0].model_name);
      })
      .catch(() => setError("Could not connect to API. It may be cold-starting — try again in 30s."));
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

      <main className="max-w-5xl mx-auto px-4 py-8">
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
              <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-xl text-sm">
                {error}
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
      </main>

      <footer className="text-center py-6 text-gray-400 text-sm">
        Crop Disease Detection &mdash; Powered by PyTorch &amp; FastAPI
      </footer>
    </div>
  );
}

export default App;
