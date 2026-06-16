# AgriBot — Precision Agriculture AI System
## Complete Setup Guide

---

## Prerequisites
- Python 3.10+
- Node.js (optional, for asset bundling)
- [Ollama](https://ollama.ai) installed locally (for Module 3 chat)
- NVIDIA GPU recommended for TensorFlow training (CPU works but is slow)

---

## Step 1 — Project setup

```bash
# Clone / unzip project, then:
cd agribot_project

# Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Apply database migrations
python manage.py makemigrations core
python manage.py migrate

# Create admin user
python manage.py createsuperuser

# Collect static files
python manage.py collectstatic --noinput
```

---

## Step 2 — Train Module 1 (Crop Recommendation)

```bash
# Download dataset from Kaggle:
# https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
# Place CSV at: data/Crop_recommendation.csv

python crop_recommendation/train_model.py
# Output: core/ml_models/crop_rf_model.pkl
#         core/ml_models/label_encoder.pkl
```

---

## Step 3 — Train Module 2 (Disease Detection)

```bash
# Download PlantVillage dataset:
# kaggle datasets download -d abdallahalidev/plantvillage-dataset
# Unzip to: data/plantvillage/   (sub-folders per class)

python disease_detection/train_cnn.py
# Output: core/ml_models/disease_model.h5
#         disease_detection/class_names.py
```

Expected accuracy: ~95%+ top-1, ~99%+ top-3 on PlantVillage.

---

## Step 4 — Start Ollama (Module 3 Chat)

```bash
# Install Ollama: https://ollama.ai/download

# Pull the vision model
ollama pull llama3.2-vision

# Start the Ollama server (runs on port 11434 by default)
ollama serve
```

---

## Step 5 — Run AgriBot

```bash
python manage.py runserver

# Open: http://127.0.0.1:8000
```

---

## Environment variables (optional `.env`)

```
DJANGO_SECRET_KEY=your-secret-key-here
DEBUG=True
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2-vision
```

---

## Architecture summary

```
Browser (AJAX/Fetch)
        │
        ▼
Django views.py
   ├── /api/crop/predict/     → joblib RF model → CropPrediction DB
   ├── /api/disease/predict/  → TF/Keras CNN    → DiseasePrediction DB
   └── /api/chat/             → requests → Ollama → ChatMessage DB
```

---

## Admin panel

Visit `http://127.0.0.1:8000/admin/` to browse all prediction history.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `RF model not found` | Run `train_model.py` first |
| `CNN model not found` | Run `train_cnn.py` first |
| `Cannot connect to Ollama` | Run `ollama serve` in a separate terminal |
| TF OOM error | Reduce `BATCH_SIZE` in `train_cnn.py` |
| CSRF error on API | Ensure `X-CSRFToken` header is sent (already handled in JS) |
