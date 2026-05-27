import { useCallback, useRef } from "react";
import { Upload, X, Camera } from "lucide-react";

export default function ImageUpload({ onImageSelect, imagePreview, onReset }) {
  const fileInputRef = useRef(null);

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith("image/")) {
        onImageSelect(file);
      }
    },
    [onImageSelect]
  );

  const handleDragOver = (e) => e.preventDefault();

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) onImageSelect(file);
  };

  if (imagePreview) {
    return (
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-4 relative">
        <button
          onClick={onReset}
          className="absolute top-3 right-3 bg-white/90 hover:bg-white p-1.5 rounded-full shadow-md transition-colors z-10"
        >
          <X className="w-4 h-4 text-gray-600" />
        </button>
        <img
          src={imagePreview}
          alt="Uploaded leaf"
          className="w-full h-72 object-contain rounded-xl bg-gray-50"
        />
      </div>
    );
  }

  return (
    <div
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onClick={() => fileInputRef.current?.click()}
      className="bg-white rounded-2xl shadow-sm border-2 border-dashed border-emerald-200 hover:border-emerald-400 p-12 text-center cursor-pointer transition-colors group"
    >
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept="image/jpeg,image/png,image/webp,image/bmp"
        className="hidden"
      />
      <div className="flex flex-col items-center gap-4">
        <div className="bg-emerald-50 group-hover:bg-emerald-100 p-4 rounded-2xl transition-colors">
          <Upload className="w-8 h-8 text-emerald-500" />
        </div>
        <div>
          <p className="text-gray-700 font-semibold">
            Drop a leaf image here or click to upload
          </p>
          <p className="text-gray-400 text-sm mt-1">
            JPEG, PNG, WebP — max 10MB
          </p>
        </div>
      </div>
    </div>
  );
}
