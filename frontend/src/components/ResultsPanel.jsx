import { CheckCircle, XCircle, Clock, Zap } from "lucide-react";

export default function ResultsPanel({ result, heatmap, imagePreview }) {
  if (!result) return null;

  const topPrediction = result.prediction.replace(/_/g, " ");
  const confidence = (result.confidence * 100).toFixed(1);
  const isHealthy = result.prediction.toLowerCase().includes("healthy");

  return (
    <div className="space-y-6">
      {/* Top Prediction Card */}
      <div
        className={`rounded-2xl shadow-sm border p-6 ${
          isHealthy
            ? "bg-green-50 border-green-200"
            : "bg-amber-50 border-amber-200"
        }`}
      >
        <div className="flex items-start gap-4">
          <div
            className={`p-2 rounded-xl ${
              isHealthy ? "bg-green-100" : "bg-amber-100"
            }`}
          >
            {isHealthy ? (
              <CheckCircle className="w-6 h-6 text-green-600" />
            ) : (
              <XCircle className="w-6 h-6 text-amber-600" />
            )}
          </div>
          <div className="flex-1">
            <p className="text-sm font-medium text-gray-500 mb-1">
              Diagnosis
            </p>
            <h2 className="text-xl font-bold text-gray-900">{topPrediction}</h2>
            <div className="mt-3 flex items-center gap-4">
              <div className="flex items-center gap-1.5">
                <Zap className="w-4 h-4 text-gray-400" />
                <span className="text-sm text-gray-600">
                  {confidence}% confidence
                </span>
              </div>
              <div className="flex items-center gap-1.5">
                <Clock className="w-4 h-4 text-gray-400" />
                <span className="text-sm text-gray-600">
                  {result.latency_ms?.toFixed(0)}ms
                </span>
              </div>
            </div>
            {/* Confidence Bar */}
            <div className="mt-3 w-full bg-gray-200 rounded-full h-2.5">
              <div
                className={`h-2.5 rounded-full transition-all duration-500 ${
                  isHealthy ? "bg-green-500" : "bg-amber-500"
                }`}
                style={{ width: `${confidence}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Model Info */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 px-6 py-3">
        <p className="text-sm text-gray-500">
          Model: <span className="font-semibold text-gray-700">{result.model_used}</span>
        </p>
      </div>

      {/* Top-K Predictions */}
      {result.top_k && result.top_k.length > 1 && (
        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
          <h3 className="text-sm font-semibold text-gray-700 mb-4">
            Top Predictions
          </h3>
          <div className="space-y-3">
            {result.top_k.map((item, idx) => {
              const pct = (item.confidence * 100).toFixed(1);
              const label = item.class.replace(/_/g, " ");
              return (
                <div key={idx}>
                  <div className="flex justify-between text-sm mb-1">
                    <span
                      className={`${
                        idx === 0 ? "font-semibold text-gray-900" : "text-gray-600"
                      }`}
                    >
                      {label}
                    </span>
                    <span className="text-gray-500 font-mono">{pct}%</span>
                  </div>
                  <div className="w-full bg-gray-100 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full transition-all duration-500 ${
                        idx === 0 ? "bg-emerald-500" : "bg-gray-300"
                      }`}
                      style={{ width: `${Math.max(pct, 1)}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* GradCAM Heatmap */}
      {heatmap && (
        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
          <h3 className="text-sm font-semibold text-gray-700 mb-4">
            GradCAM Explanation
          </h3>
          <p className="text-xs text-gray-400 mb-3">
            Highlighted regions show where the model focused to make its prediction
          </p>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-xs text-gray-500 mb-2 text-center">Original</p>
              <img
                src={imagePreview}
                alt="Original"
                className="w-full rounded-xl object-contain bg-gray-50 h-48"
              />
            </div>
            <div>
              <p className="text-xs text-gray-500 mb-2 text-center">GradCAM Heatmap</p>
              <img
                src={`data:image/png;base64,${heatmap}`}
                alt="GradCAM heatmap"
                className="w-full rounded-xl object-contain bg-gray-50 h-48"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
