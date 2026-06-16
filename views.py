"""
views.py — AgriBot (Clean Core)
=================================
Module 1: Random Forest crop recommendation
Module 2: ViT-Small/16 disease detection (plain inference, no thresholding)
Module 3: Ollama/Moondream English-only chat

Removed:
  ✗ 3-Tier OOD / Soft Thresholding
  ✗ Grad-CAM heatmap generation
  ✗ Multilingual (Hindi / Gujarati / Hinglish)

Kept:
  ✓ All variable names (RF_MODEL, VIT_MODEL, OLLAMA_URL, DISEASE_CLASS_NAMES …)
  ✓ All DB model interactions (CropPrediction, DiseasePrediction, ChatMessage)
  ✓ All imports that are still needed
"""

import io
import os
import json
import uuid
import pickle
import logging
import traceback

import numpy as np
from PIL import Image

from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

import requests

from .models import CropPrediction, DiseasePrediction, ChatMessage
from disease_detection.class_names import DISEASE_CLASS_NAMES

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────

_MODEL_DIR = os.path.join(settings.BASE_DIR, "core", "ml_models")
_VIT_DIR   = os.path.join(_MODEL_DIR, "vit_disease_model", "best_model")

OLLAMA_URL   = getattr(settings, "OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL = getattr(settings, "OLLAMA_MODEL", "moondream")

# ─────────────────────────────────────────────────────────────
#  1.  MODEL LOADING
# ─────────────────────────────────────────────────────────────

RF_MODEL      = None
LABEL_ENCODER = None
VIT_MODEL     = None
VIT_PROCESSOR = None
VIT_DEVICE    = None


def _load_rf():
    path    = os.path.join(_MODEL_DIR, "crop_rf_model.pkl")
    le_path = os.path.join(_MODEL_DIR, "label_encoder.pkl")
    if not os.path.exists(path):    raise FileNotFoundError(path)
    if not os.path.exists(le_path): raise FileNotFoundError(le_path)
    try:
        import joblib
        return joblib.load(path), joblib.load(le_path)
    except Exception:
        with open(path,    "rb") as f: model = pickle.load(f)
        with open(le_path, "rb") as f: le    = pickle.load(f)
        return model, le


def _load_vit():
    """Load ViT-Small from saved checkpoint. Stays in eval() at all times."""
    import torch
    from transformers import ViTForImageClassification, ViTImageProcessor

    if not os.path.exists(_VIT_DIR):
        raise FileNotFoundError(f"ViT model not found: {_VIT_DIR}")

    try:
        processor = ViTImageProcessor.from_pretrained(_VIT_DIR)
        logger.info("Processor loaded from saved dir")
    except Exception:
        processor = ViTImageProcessor.from_pretrained(
            "WinKawaks/vit-small-patch16-224"
        )
        logger.info("Processor loaded from HuggingFace")

    # Enforce correct ViT normalisation: mean = std = [0.5, 0.5, 0.5]
    processor.image_mean = [0.5, 0.5, 0.5]
    processor.image_std  = [0.5, 0.5, 0.5]
    processor.size       = {"height": 224, "width": 224}

    model = ViTForImageClassification.from_pretrained(
        _VIT_DIR,
        local_files_only=True,
        ignore_mismatched_sizes=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    model.eval()

    logger.info("✅  ViT loaded | device=%s | classes=%d",
                device, model.config.num_labels)
    return model, processor, device


# ── Load both models at Django startup ───────────────────────
try:
    RF_MODEL, LABEL_ENCODER = _load_rf()
    logger.info("✅  RF Crop model ready")
except Exception as exc:
    logger.warning("⚠️  RF model not ready: %s", exc)

try:
    VIT_MODEL, VIT_PROCESSOR, VIT_DEVICE = _load_vit()
    logger.info("✅  ViT Disease model ready")
except Exception as exc:
    logger.error("❌  ViT load FAILED: %s\n%s", exc, traceback.format_exc())


# ─────────────────────────────────────────────────────────────
#  2.  ViT INFERENCE HELPER
# ─────────────────────────────────────────────────────────────

def _run_vit_inference(image_bytes: bytes):
    """
    Run ViT forward pass on raw image bytes.

    Returns:
        top3_idx : np.ndarray (3,)  — top-3 class indices, descending prob
        probs_np : np.ndarray (38,) — full softmax probability vector
    """
    import torch
    import torch.nn.functional as F

    img    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = VIT_PROCESSOR(
        images=img,
        return_tensors="pt",
        do_resize=True,
        do_normalize=True,
    )
    pixel_values = inputs["pixel_values"].to(VIT_DEVICE)

    with torch.no_grad():
        outputs = VIT_MODEL(pixel_values=pixel_values)
        probs   = F.softmax(outputs.logits, dim=-1)[0]   # (38,)

    probs_np = probs.cpu().numpy()
    top3_idx = np.argsort(probs_np)[::-1][:3]
    return top3_idx, probs_np


# ─────────────────────────────────────────────────────────────
#  3.  OLLAMA HELPER — ENGLISH ONLY
# ─────────────────────────────────────────────────────────────

AGRIBOT_SYSTEM_PROMPT = (
    "You are AgriBot, an expert agricultural AI assistant. "
    "Help farmers with plant disease diagnosis, treatment recommendations, "
    "soil health, irrigation, and crop management. "
    "Always respond in English only. "
    "Structure your answer with: "
    "1) Brief diagnosis, "
    "2) Immediate action steps, "
    "3) Organic and chemical treatment options, "
    "4) Prevention tips. "
    "Keep the response under 200 words. Be practical and farmer-friendly."
)

OFFLINE_REPLY = (
    "🔌 **AgriBot is offline** — Ollama is not running.\n\n"
    "To fix this:\n"
    "1. Open a new terminal\n"
    "2. Run: `ollama serve`\n"
    "3. Ensure moondream is pulled: `ollama pull moondream`\n"
    "4. Refresh this page"
)


def _call_ollama(prompt: str) -> str:
    """
    Call Ollama for a single English agricultural advisory response.
    Tries /api/generate first (moondream native), falls back to /api/chat.
    Returns the response string or empty string on failure.
    """
    full_prompt = f"{AGRIBOT_SYSTEM_PROMPT}\n\nFarmer: {prompt}\n\nAgriBot:"

    # Primary: /api/generate
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model":   OLLAMA_MODEL,
                "prompt":  full_prompt,
                "stream":  False,
                "options": {
                    "temperature": 0.6,
                    "num_predict": 400,
                    "stop": ["Farmer:", "Human:", "User:"],
                },
            },
            timeout=90,
        )
        resp.raise_for_status()
        result = resp.json().get("response", "").strip()
        if result:
            return result
    except Exception as e:
        logger.warning("⚠️  /api/generate failed (%s), trying /api/chat...", e)

    # Fallback: /api/chat
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": AGRIBOT_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                "stream":  False,
                "options": {"temperature": 0.6, "num_predict": 400},
            },
            timeout=90,
        )
        resp.raise_for_status()
        data = resp.json()
        return (
            data.get("message", {}).get("content", "")
            or data.get("response", "")
        ).strip()
    except Exception as e:
        logger.warning("⚠️  /api/chat also failed: %s", e)
        return ""


# ─────────────────────────────────────────────────────────────
#  4.  PAGE VIEWS
# ─────────────────────────────────────────────────────────────

def dashboard(request):
    ctx = {
        "crop_count":      CropPrediction.objects.count(),
        "disease_count":   DiseasePrediction.objects.count(),
        "chat_count":      ChatMessage.objects.filter(role="user").count(),
        "recent_crops":    CropPrediction.objects.all()[:5],
        "recent_diseases": DiseasePrediction.objects.all()[:5],
        "vit_loaded":      VIT_MODEL is not None,
        "rf_loaded":       RF_MODEL  is not None,
    }
    return render(request, "core/dashboard.html", ctx)


def crop_page(request):
    ctx = {
        "example_crops": ["rice", "wheat", "maize", "mango", "banana"],
        "inputs": [
            ("N",           "Nitrogen",    "mg/kg", "e.g. 40"),
            ("P",           "Phosphorus",  "mg/kg", "e.g. 30"),
            ("K",           "Potassium",   "mg/kg", "e.g. 50"),
            ("temperature", "Temperature", "°C",    "e.g. 25"),
            ("humidity",    "Humidity",    "%",      "e.g. 70"),
            ("ph",          "Soil pH",     "pH",    "e.g. 6.5"),
            ("rainfall",    "Rainfall",    "mm",    "e.g. 100"),
        ],
    }
    return render(request, "core/crop.html", ctx)


def disease_page(request):
    ctx = {
        "model_type":   "ViT-Small/16 (~98% acc)" if VIT_MODEL else "Not loaded",
        "model_loaded": VIT_MODEL is not None,
    }
    return render(request, "core/disease.html", ctx)


def agribot_page(request):
    ctx = {
        "quick_prompts": [
            "What causes tomato late blight?",
            "Organic remedy for powdery mildew?",
            "How to improve soil nitrogen?",
            "Best fertiliser schedule for rice?",
            "How to prevent leaf scorch in strawberries?",
            "What is Cercospora leaf spot?",
        ]
    }
    return render(request, "core/agribot.html", ctx)


def crop_history(request):
    return render(request, "core/history_crop.html",
                  {"predictions": CropPrediction.objects.all()[:100]})


def disease_history(request):
    return render(request, "core/history_disease.html",
                  {"predictions": DiseasePrediction.objects.all()[:100]})


# ─────────────────────────────────────────────────────────────
#  5.  API: CROP RECOMMENDATION
# ─────────────────────────────────────────────────────────────

@csrf_exempt
@require_http_methods(["POST"])
def api_crop_predict(request):
    if RF_MODEL is None or LABEL_ENCODER is None:
        return JsonResponse(
            {"error": "Crop model not loaded. Run fix_and_train.py first."},
            status=503,
        )
    try:
        body    = json.loads(request.body)
        fields  = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        missing = [f for f in fields if f not in body]
        if missing:
            return JsonResponse({"error": f"Missing fields: {missing}"}, status=400)

        features   = np.array([[float(body[f]) for f in fields]])
        proba      = RF_MODEL.predict_proba(features)[0]
        top3_idx   = np.argsort(proba)[::-1][:3]
        pred_label = LABEL_ENCODER.inverse_transform([top3_idx[0]])[0]
        confidence = float(proba[top3_idx[0]])

        top3 = [
            {
                "crop":       LABEL_ENCODER.inverse_transform([i])[0],
                "confidence": round(float(proba[i]) * 100, 2),
            }
            for i in top3_idx
        ]

        session_id = request.session.get("agribot_session") or str(uuid.uuid4())
        request.session["agribot_session"] = session_id

        CropPrediction.objects.create(
            nitrogen=body["N"],        phosphorus=body["P"],
            potassium=body["K"],       temperature=body["temperature"],
            humidity=body["humidity"], ph=body["ph"],
            rainfall=body["rainfall"],
            predicted_crop=str(pred_label),
            confidence_score=confidence,
            session_id=session_id,
        )

        return JsonResponse({
            "crop":       str(pred_label),
            "confidence": round(confidence * 100, 2),
            "top_3":      top3,
        })

    except (ValueError, KeyError) as exc:
        return JsonResponse({"error": f"Invalid input: {exc}"}, status=400)
    except Exception:
        logger.error(traceback.format_exc())
        return JsonResponse({"error": "Internal server error."}, status=500)


# ─────────────────────────────────────────────────────────────
#  6.  API: DISEASE DETECTION
#       Plain ViT inference — no thresholding, no Grad-CAM
# ─────────────────────────────────────────────────────────────

@csrf_exempt
@require_http_methods(["POST"])
def api_disease_predict(request):
    if VIT_MODEL is None:
        return JsonResponse(
            {"error": "ViT model not loaded. "
                      "Check core/ml_models/vit_disease_model/best_model/"},
            status=503,
        )

    if "image" not in request.FILES:
        return JsonResponse({"error": "No image file provided."}, status=400)

    try:
        image_file  = request.FILES["image"]
        session_id  = request.session.get("agribot_session") or str(uuid.uuid4())
        request.session["agribot_session"] = session_id

        # Read image bytes for inference
        image_bytes = image_file.read()

        # ── ViT Inference ─────────────────────────────────────────────
        top3_idx, probs = _run_vit_inference(image_bytes)

        pred_idx   = int(top3_idx[0])
        pred_label = DISEASE_CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])
        conf_pct   = round(confidence * 100, 2)

        top3 = [
            {
                "disease":    DISEASE_CLASS_NAMES[int(i)],
                "confidence": round(float(probs[i]) * 100, 2),
            }
            for i in top3_idx
        ]

        # ── Save to database ──────────────────────────────────────────
        image_file.seek(0)
        pred_record = DiseasePrediction(session_id=session_id, image=image_file)
        pred_record.predicted_disease = pred_label
        pred_record.confidence_score  = confidence
        pred_record.top_3_predictions = top3
        pred_record.save()   # auto-parses plant_name, disease_name, is_healthy

        logger.info("✅  Disease: %s | conf=%.1f%%", pred_label, conf_pct)

        return JsonResponse({
            "id":         pred_record.pk,
            "disease":    pred_record.disease_name,
            "plant":      pred_record.plant_name,
            "is_healthy": pred_record.is_healthy,
            "confidence": conf_pct,
            "top_3":      top3,
            "model_used": "ViT-Small/16",
        })

    except Exception:
        logger.error("Disease prediction error:\n%s", traceback.format_exc())
        return JsonResponse({"error": "Internal server error."}, status=500)


# ─────────────────────────────────────────────────────────────
#  7.  API: AGRIBOT CHAT — ENGLISH ONLY
# ─────────────────────────────────────────────────────────────

@csrf_exempt
@require_http_methods(["POST"])
def api_chat(request):
    """
    Single English response from Ollama/Moondream.
    No multilingual translation — fast and stable.
    """
    try:
        body     = json.loads(request.body)
        user_msg = body.get("message", "").strip()
        if not user_msg:
            return JsonResponse({"error": "Empty message."}, status=400)

        session_id = body.get("session_id") or str(uuid.uuid4())
        disease_id = body.get("disease_id")

        # ── Optional disease context from Disease Detection ───────────
        disease_obj    = None
        context_prefix = ""
        if disease_id:
            try:
                disease_obj = DiseasePrediction.objects.get(pk=disease_id)
                context_prefix = (
                    f"Context: The farmer's leaf scan detected "
                    f"'{disease_obj.predicted_disease}' "
                    f"at {disease_obj.confidence_score:.1%} confidence "
                    f"on {disease_obj.plant_name}. "
                    f"Provide targeted treatment advice. "
                )
            except DiseasePrediction.DoesNotExist:
                pass

        # ── Persist user message ──────────────────────────────────────
        ChatMessage.objects.create(
            session_id=session_id,
            role="user",
            content=user_msg,
            related_disease=disease_obj,
        )

        # ── Ollama health check ───────────────────────────────────────
        try:
            ping = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
            ollama_online = (ping.status_code == 200)
        except Exception:
            ollama_online = False

        if not ollama_online:
            ChatMessage.objects.create(
                session_id=session_id,
                role="assistant",
                content=OFFLINE_REPLY,
                model_used="offline",
                related_disease=disease_obj,
            )
            return JsonResponse({
                "reply":      OFFLINE_REPLY,
                "session_id": session_id,
                "model":      "offline",
            })

        # ── Call Ollama ───────────────────────────────────────────────
        full_prompt = context_prefix + user_msg
        reply       = _call_ollama(full_prompt)

        if not reply:
            reply = (
                "I could not generate a response right now. "
                "Please try rephrasing your question."
            )

        # ── Persist assistant reply ───────────────────────────────────
        ChatMessage.objects.create(
            session_id=session_id,
            role="assistant",
            content=reply,
            model_used=OLLAMA_MODEL,
            related_disease=disease_obj,
        )

        return JsonResponse({
            "reply":      reply,
            "session_id": session_id,
            "model":      OLLAMA_MODEL,
        })

    except requests.exceptions.Timeout:
        return JsonResponse(
            {"error": "⏱ Ollama timed out. Wait a few seconds and try again."},
            status=504,
        )
    except requests.exceptions.ConnectionError:
        return JsonResponse(
            {"error": "🔌 Cannot connect to Ollama. Run: ollama serve"},
            status=503,
        )
    except Exception:
        logger.error("Chat error:\n%s", traceback.format_exc())
        return JsonResponse({"error": "Internal server error."}, status=500)
