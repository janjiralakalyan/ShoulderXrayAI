"""
FastAPI backend for Shoulder X-ray AI Analysis.
Provides a /predict endpoint that accepts X-ray images and returns
prediction, confidence, Grad-CAM heatmap, and a medical report.
"""

import os
import logging
from contextlib import asynccontextmanager

import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import base64

from backend.utils import (
    preprocess_image,
    generate_gradcam,
    overlay_heatmap,
    generate_report,
    generate_pdf_report,
    image_to_base64,
)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "model.h5"),
)
CONFIDENCE_THRESHOLD = 0.5

logger = logging.getLogger("shoulder-xray-api")
logging.basicConfig(level=logging.INFO)

# ─────────────────────────────────────────────
# Model Loading (lifespan)
# ─────────────────────────────────────────────

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at startup."""
    global model
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at {MODEL_PATH}")
        logger.error(
            "Please run the training script first or set MODEL_PATH env var."
        )
        # Allow the app to start anyway for development; /predict will 503
    else:
        logger.info(f"Loading model from {MODEL_PATH} ...")
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded successfully.")
    yield
    logger.info("Shutting down.")


# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────

app = FastAPI(
    title="Shoulder X-ray AI Analysis API",
    description=(
        "Upload a shoulder X-ray image and receive an AI-powered prediction "
        "(NORMAL / ABNORMAL), confidence score, Grad-CAM heatmap, and a "
        "basic medical report."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow CORS for the Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


# ─────────────────────────────────────────────
# Prediction Endpoint
# ─────────────────────────────────────────────

@app.post("/predict", tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Analyze a shoulder X-ray image.

    - **file**: An X-ray image (JPEG / PNG)

    Returns prediction, confidence, Grad-CAM heatmap (base64), and a report.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not loaded. Please ensure 'shoulder_xray_model.h5' "
                "exists in the backend directory or set MODEL_PATH."
            ),
        )

    # Validate file type
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a JPEG or PNG image.",
        )

    try:
        # Read the image bytes
        image_bytes = await file.read()

        # Preprocess
        img_array, original_image = preprocess_image(image_bytes)

        # Predict
        prediction_prob = float(model.predict(img_array, verbose=0)[0][0])
        prediction_label = "ABNORMAL" if prediction_prob >= CONFIDENCE_THRESHOLD else "NORMAL"
        confidence = prediction_prob if prediction_label == "ABNORMAL" else 1 - prediction_prob

        # Grad-CAM
        heatmap = generate_gradcam(model, img_array)
        overlay_image = overlay_heatmap(heatmap, original_image)
        heatmap_base64 = image_to_base64(overlay_image)

        # Report
        report = generate_report(prediction_label, confidence)

        # PDF Report
        pdf_bytes = generate_pdf_report(
            prediction_label,
            confidence,
            original_image,
            overlay_image,
            report
        )
        pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')

        logger.info(
            f"Prediction: {prediction_label} | Confidence: {confidence:.2%}"
        )

        return JSONResponse(
            content={
                "prediction": prediction_label,
                "confidence": round(confidence, 4),
                "heatmap": heatmap_base64,
                "report": report,
                "pdf": pdf_base64,
            }
        )

    except Exception as e:
        logger.exception("Error during prediction")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# Run with: uvicorn backend.main:app --reload
# ─────────────────────────────────────────────
