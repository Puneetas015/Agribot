from django.db import models
from django.utils import timezone


# ─────────────────────────────────────────────────────────────
#  Module 1 – Crop Recommendation History
# ─────────────────────────────────────────────────────────────
class CropPrediction(models.Model):
    """Stores every crop recommendation request and its result."""

    # Soil & climate inputs
    nitrogen      = models.FloatField(help_text="Nitrogen (N) ratio in soil")
    phosphorus    = models.FloatField(help_text="Phosphorus (P) ratio in soil")
    potassium     = models.FloatField(help_text="Potassium (K) ratio in soil")
    temperature   = models.FloatField(help_text="Temperature in °C")
    humidity      = models.FloatField(help_text="Relative humidity (%)")
    ph            = models.FloatField(help_text="Soil pH value")
    rainfall      = models.FloatField(help_text="Rainfall in mm")

    # Model output
    predicted_crop        = models.CharField(max_length=100)
    confidence_score      = models.FloatField(
        null=True, blank=True,
        help_text="Max class probability from Random Forest (0-1)"
    )

    # Metadata
    created_at = models.DateTimeField(default=timezone.now)
    session_id = models.CharField(max_length=64, blank=True, null=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name        = "Crop Prediction"
        verbose_name_plural = "Crop Predictions"

    def __str__(self):
        return f"{self.predicted_crop} | N={self.nitrogen} P={self.phosphorus} K={self.potassium} | {self.created_at:%Y-%m-%d %H:%M}"


# ─────────────────────────────────────────────────────────────
#  Module 2 – Plant Disease Detection History
# ─────────────────────────────────────────────────────────────
class DiseasePrediction(models.Model):
    """Stores every uploaded leaf image and its disease diagnosis."""

    # Uploaded image (stored in media/disease_images/)
    image = models.ImageField(
        upload_to="disease_images/%Y/%m/",
        help_text="Uploaded plant leaf image"
    )

    # Model output
    predicted_disease   = models.CharField(max_length=200)
    confidence_score    = models.FloatField(
        null=True, blank=True,
        help_text="Softmax probability for the top predicted class (0-1)"
    )
    top_3_predictions   = models.JSONField(
        null=True, blank=True,
        help_text="List of {label, confidence} for the top-3 classes"
    )

    # Plant metadata parsed from PlantVillage label  (e.g. "Tomato__Late_blight")
    plant_name    = models.CharField(max_length=100, blank=True)
    disease_name  = models.CharField(max_length=100, blank=True)
    is_healthy    = models.BooleanField(default=False)

    # Metadata
    created_at = models.DateTimeField(default=timezone.now)
    session_id = models.CharField(max_length=64, blank=True, null=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name        = "Disease Prediction"
        verbose_name_plural = "Disease Predictions"

    def save(self, *args, **kwargs):
        """Auto-parse plant_name / disease_name from the PlantVillage label."""
        if self.predicted_disease and not self.plant_name:
            parts = self.predicted_disease.replace("___", "__").split("__")
            self.plant_name  = parts[0].replace("_", " ").title() if len(parts) > 0 else ""
            self.disease_name = parts[1].replace("_", " ").title() if len(parts) > 1 else ""
            self.is_healthy  = "healthy" in self.disease_name.lower()
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.predicted_disease} ({self.confidence_score:.2%}) | {self.created_at:%Y-%m-%d %H:%M}"


# ─────────────────────────────────────────────────────────────
#  Module 3 – AgriBot Chat History
# ─────────────────────────────────────────────────────────────
class ChatMessage(models.Model):
    """Persists every turn of an AgriBot conversation for review / export."""

    ROLE_CHOICES = [
        ("user",      "User"),
        ("assistant", "Assistant (AgriBot)"),
        ("system",    "System"),
    ]

    session_id  = models.CharField(max_length=64, db_index=True)
    role        = models.CharField(max_length=16, choices=ROLE_CHOICES)
    content     = models.TextField()
    model_used  = models.CharField(
        max_length=100, blank=True,
        help_text="LLM model tag returned by Ollama (e.g. moondream)"
    )

    # Optional link to the disease prediction that triggered this conversation
    related_disease = models.ForeignKey(
        DiseasePrediction,
        null=True, blank=True,
        on_delete=models.SET_NULL,
        related_name="chat_messages"
    )

    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["session_id", "created_at"]
        verbose_name        = "Chat Message"
        verbose_name_plural = "Chat Messages"

    def __str__(self):
        snippet = self.content[:60] + ("…" if len(self.content) > 60 else "")
        return f"[{self.role.upper()}] {snippet}"


# ─────────────────────────────────────────────────────────────
#  Optional: track which crops appear most often (analytics)
# ─────────────────────────────────────────────────────────────
class PredictionSummary(models.Model):
    """Aggregated stats refreshed periodically (e.g. via a management command)."""

    date              = models.DateField(unique=True)
    total_crop_preds  = models.PositiveIntegerField(default=0)
    total_disease_preds = models.PositiveIntegerField(default=0)
    most_predicted_crop = models.CharField(max_length=100, blank=True)
    most_detected_disease = models.CharField(max_length=200, blank=True)

    class Meta:
        ordering = ["-date"]

    def __str__(self):
        return f"Summary {self.date} — crops={self.total_crop_preds} diseases={self.total_disease_preds}"
