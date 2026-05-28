import { Leaf } from "lucide-react";

export default function Header() {
  return (
    <header className="bg-white/80 backdrop-blur-sm border-b border-gray-100 sticky top-0 z-10">
      <div className="max-w-5xl mx-auto px-4 py-4 flex items-center gap-3">
        <div className="bg-emerald-100 p-2 rounded-xl">
          <Leaf className="w-6 h-6 text-emerald-600" />
        </div>
        <div>
          <h1 className="text-xl font-bold text-gray-900 leading-tight">
            Crop Disease Detection
          </h1>
          <p className="text-sm text-gray-500">
            AI-powered plant disease diagnosis with explainability
          </p>
        </div>
      </div>
    </header>
  );
}
