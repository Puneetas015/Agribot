"""
core/urls.py — AgriBot App URL Routes
"""
from django.urls import path
from . import views

urlpatterns = [
    # ── Page views ──────────────────────────────────────────
    path("",                     views.dashboard,          name="dashboard"),
    path("crop/",                views.crop_page,           name="crop"),
    path("disease/",             views.disease_page,        name="disease"),
    path("agribot/",             views.agribot_page,        name="agribot"),
    path("history/crop/",        views.crop_history,        name="crop_history"),
    path("history/disease/",     views.disease_history,     name="disease_history"),

    # ── AJAX / Fetch API endpoints ──────────────────────────
    path("api/crop/predict/",    views.api_crop_predict,    name="api_crop_predict"),
    path("api/disease/predict/", views.api_disease_predict, name="api_disease_predict"),
    path("api/chat/",            views.api_chat,            name="api_chat"),
]
