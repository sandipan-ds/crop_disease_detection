import axios from "axios";

const API_BASE = "https://crop-disease-api-1049249498032.us-central1.run.app";

const api = axios.create({
  baseURL: API_BASE,
  timeout: 120000, // 2 min (cold start + model loading)
});

export async function fetchModels() {
  const { data } = await api.get("/models");
  return data.models;
}

export async function predict(imageFile, modelName, topK = 5) {
  const formData = new FormData();
  formData.append("image", imageFile);
  formData.append("model_name", modelName);
  formData.append("top_k", topK);

  const { data } = await api.post("/predict", formData);
  return data;
}

export async function explain(imageFile, modelName) {
  const formData = new FormData();
  formData.append("image", imageFile);
  formData.append("model_name", modelName);

  const { data } = await api.post("/explain", formData);
  return data;
}

export async function healthCheck() {
  const { data } = await api.get("/health");
  return data;
}
