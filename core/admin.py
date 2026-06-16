"""
core/admin.py — Register AgriBot models in Django admin panel.
Visit: http://127.0.0.1:8000/admin/
"""
from django.contrib import admin
from .models import CropPrediction, DiseasePrediction, ChatMessage, PredictionSummary


@admin.register(CropPrediction)
class CropPredictionAdmin(admin.ModelAdmin):
    list_display  = ("predicted_crop", "confidence_score", "nitrogen", "phosphorus",
                     "potassium", "temperature", "humidity", "ph", "rainfall", "created_at")
    list_filter   = ("predicted_crop",)
    search_fields = ("predicted_crop",)
    readonly_fields = ("created_at",)
    ordering      = ("-created_at",)


@admin.register(DiseasePrediction)
class DiseasePredictionAdmin(admin.ModelAdmin):
    list_display  = ("plant_name", "disease_name", "is_healthy", "confidence_score", "created_at")
    list_filter   = ("is_healthy", "plant_name")
    search_fields = ("plant_name", "disease_name", "predicted_disease")
    readonly_fields = ("created_at", "plant_name", "disease_name", "is_healthy")
    ordering      = ("-created_at",)


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display  = ("role", "session_id", "model_used", "short_content", "created_at")
    list_filter   = ("role",)
    search_fields = ("session_id", "content")
    ordering      = ("-created_at",)

    def short_content(self, obj):
        return obj.content[:80] + ("…" if len(obj.content) > 80 else "")
    short_content.short_description = "Content preview"


@admin.register(PredictionSummary)
class PredictionSummaryAdmin(admin.ModelAdmin):
    list_display = ("date", "total_crop_preds", "total_disease_preds",
                    "most_predicted_crop", "most_detected_disease")
    ordering     = ("-date",)
