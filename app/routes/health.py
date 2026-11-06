"""Health and model-info endpoints."""

from flask import Blueprint, jsonify

from app.core.logging_config import get_logger
from src.ml.model import model_manager

logger = get_logger("routes.health")
bp = Blueprint("health", __name__)


@bp.get("/health")
def health():
    return jsonify({"status": "healthy"}), 200


@bp.get("/model/info")
def model_info():
    return jsonify({"success": True, "data": model_manager.model_info()}), 200