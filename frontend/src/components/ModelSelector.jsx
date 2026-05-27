import { Cpu } from "lucide-react";

export default function ModelSelector({ models, selected, onChange }) {
  return (
    <div>
      <label className="flex items-center gap-2 text-sm font-semibold text-gray-700 mb-2">
        <Cpu className="w-4 h-4" />
        Select Model
      </label>
      <select
        value={selected}
        onChange={(e) => onChange(e.target.value)}
        className="w-full bg-gray-50 border border-gray-200 rounded-xl px-4 py-3 text-gray-800 font-medium focus:outline-none focus:ring-2 focus:ring-emerald-400 focus:border-transparent transition-all"
      >
        {models.map((m) => (
          <option key={m.model_name} value={m.model_name}>
            {m.display_name} — {m.type}
            {m.val_f1_macro ? ` (F1: ${(m.val_f1_macro * 100).toFixed(1)}%)` : ""}
          </option>
        ))}
      </select>
    </div>
  );
}
