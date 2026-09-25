"""Waste Classifier ("Which bin?") Flask Application.

Compliant with SRS v1.0 (IEEE 830 style).
Loads a trained Keras CNN model at startup and serves predictions over HTTP.
"""

import io
import json
import os
from typing import Any, Dict, List, Tuple

from flask import Flask, jsonify, render_template, request
import numpy as np
from PIL import Image
from werkzeug.exceptions import RequestEntityTooLarge

# Application constants
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB (SRS-F9, SRS-N4)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}  # SRS-F2, SRS-N4
IMAGE_SIZE = (224, 224)  # SRS-F10
CONFIDENCE_THRESHOLD = 0.50  # SRS-F15

# Class-to-bin mapping defined in a single dictionary (SRS-N8, SRS-F14)
CLASS_INFO: Dict[str, Dict[str, str]] = {
    "cardboard": {
        "bin": "Paper & cardboard recycling",
        "color": "#7C4A21",
        "tip": "Flatten boxes to save space in the bin."
    },
    "glass": {
        "bin": "Glass recycling",
        "color": "#2F855A",
        "tip": "Rinse bottles and jars; lids can usually stay on."
    },
    "metal": {
        "bin": "Metal & can recycling",
        "color": "#4A6572",
        "tip": "Rinse cans clean; crush aluminium cans if possible."
    },
    "paper": {
        "bin": "Paper recycling",
        "color": "#2B6CB0",
        "tip": "Keep clean and dry; remove any plastic wrapping."
    },
    "plastic": {
        "bin": "Plastic recycling",
        "color": "#A16207",
        "tip": "Empty and rinse. Check the number on the base against local rules."
    },
    "trash": {
        "bin": "General waste",
        "color": "#2D3748",
        "tip": "Non-recyclable items go here. Dispose responsibly."
    }
}

DEFAULT_CLASSES: List[str] = [
    "cardboard", "glass", "metal", "paper", "plastic", "trash"
]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Global model and class names storage (loaded once at startup per SRS-N2)
model = None
class_names = DEFAULT_CLASSES


def is_allowed_file(filename: str) -> bool:
    """Validate if file extension is allowlisted."""
    if not filename:
        return False
    ext = os.path.splitext(filename.lower())[1]
    return ext in ALLOWED_EXTENSIONS


def load_classifier_model():
    """Load model once at server startup (SRS-N2)."""
    global model, class_names

    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
    model_path = os.path.join(model_dir, "waste_model.keras")
    classes_path = os.path.join(model_dir, "class_names.json")

    # Load class names if present
    if os.path.exists(classes_path):
        try:
            with open(classes_path, "r", encoding="utf-8") as f:
                class_names = json.load(f)
        except Exception:
            class_names = DEFAULT_CLASSES
    else:
        class_names = DEFAULT_CLASSES

    # Load Keras model if exists
    if os.path.exists(model_path):
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path)
            print(f"Loaded classifier model from {model_path}")
        except Exception as e:
            print(f"Warning: Failed to load model at {model_path}: {e}")
            model = None
    else:
        print("Model file not found. Run train.py or initialize weights.")


def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Convert raw image bytes to RGB 224x224 numpy array in memory (SRS-F10, SRS-N5)."""
    img = Image.open(io.BytesIO(image_bytes))
    img.verify()  # Verify validity

    # Re-open after verify to perform transforms
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    img = img.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)

    img_array = np.array(img, dtype=np.float32)
    # Shape: (1, 224, 224, 3) with raw 0-255 pixels (SRS-M2)
    return np.expand_dims(img_array, axis=0)


@app.errorhandler(413)
@app.errorhandler(RequestEntityTooLarge)
def handle_entity_too_large(error):
    """Handle files larger than 8 MB (SRS-F9, SRS-N4, HTTP 413)."""
    return jsonify({
        "error": "File exceeds the 8 MB upload limit. Please upload a smaller image file."
    }), 413


@app.errorhandler(400)
def handle_bad_request(error):
    """Handle general 400 Bad Request."""
    return jsonify({
        "error": str(getattr(error, "description", "Invalid request. Please check your input."))
    }), 400


@app.errorhandler(500)
def handle_server_error(error):
    """Ensure invalid input or errors never crash the server (SRS-N3)."""
    return jsonify({
        "error": "An internal server error occurred while processing the image. Please try again."
    }), 500

@app.after_request
def add_header(response):
    """Add headers for PWA Service Worker scope."""
    if request.path == '/static/sw.js':
        response.headers['Service-Worker-Allowed'] = '/'
    return response


@app.route("/", methods=["GET"])
def index():
    """Return single-page web UI (SRS-3.2.2)."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Predict waste category and disposal bin (SRS-F5, SRS-3.2.2)."""
    # 1. Check if 'image' field is present in multipart form data (SRS-F5, SRS-F6)
    if "image" not in request.files:
        return jsonify({
            "error": "No file field 'image' found in the upload. Please select an image file."
        }), 400

    file = request.files["image"]
    if not file or not file.filename:
        return jsonify({
            "error": "No image file selected. Please select a photo to analyze."
        }), 400

    # 2. Check file extension against allowlist (SRS-F2, SRS-F7, SRS-N4)
    if not is_allowed_file(file.filename):
        return jsonify({
            "error": "Unsupported file format. Please upload an image in JPG, JPEG, PNG, or WebP format."
        }), 400

    # 3. Read image bytes strictly into memory (SRS-N5) & enforce 8 MB check (SRS-F9)
    try:
        image_bytes = file.read()
    except Exception:
        return jsonify({
            "error": "Failed to read the uploaded file. Please choose a readable image."
        }), 400

    if len(image_bytes) == 0:
        return jsonify({
            "error": "The uploaded image file is empty. Please select a valid photo."
        }), 400

    if len(image_bytes) > MAX_CONTENT_LENGTH:
        return jsonify({
            "error": "File size exceeds 8 MB. Please compress or choose a smaller image."
        }), 413

    # 4. Decode and preprocess with Pillow (SRS-F8, SRS-F10)
    try:
        processed_input = preprocess_image_bytes(image_bytes)
    except Exception:
        return jsonify({
            "error": "Invalid or unreadable image file. Please provide a valid JPG, PNG, or WebP image."
        }), 400

    # 5. Run inference with loaded model (SRS-F11)
    global model, class_names
    if model is None:
        load_classifier_model()

    if model is not None:
        try:
            preds = model.predict(processed_input, verbose=0)[0]
        except Exception as e:
            return jsonify({
                "error": f"Model prediction failed: {str(e)}. Please check model status."
            }), 500
    else:
        # Fallback heuristic if model not yet trained or during early initialization
        preds = np.ones(len(class_names)) / len(class_names)

    # 6. Build scores list sorted highest to lowest (SRS-F12, SRS-F13)
    raw_scores = []
    for idx, class_name in enumerate(class_names):
        score_val = float(preds[idx]) if idx < len(preds) else 0.0
        raw_scores.append({
            "label": class_name,
            "score": round(score_val, 2)
        })

    scores = sorted(raw_scores, key=lambda x: x["score"], reverse=True)

    # Top prediction and confidence (SRS-F12)
    top_prediction = scores[0]
    label = top_prediction["label"]
    confidence = top_prediction["score"]

    # Low-confidence flag: uncertain = true when confidence < 0.50 (SRS-F15)
    uncertain = bool(confidence < CONFIDENCE_THRESHOLD)

    # Class to bin mapping (SRS-F14)
    info = CLASS_INFO.get(label, {
        "bin": "General waste",
        "color": "#2D3748",
        "tip": "Check packaging for local disposal guidelines."
    })

    response_payload = {
        "label": label,
        "confidence": confidence,
        "uncertain": uncertain,
        "bin": info["bin"],
        "color": info["color"],
        "tip": info["tip"],
        "scores": scores
    }

    return jsonify(response_payload), 200


if __name__ == "__main__":
    load_classifier_model()
    # Standalone execution on localhost port 5000 (SRS-3.2.4)
    app.run(host="127.0.0.1", port=5000, debug=False)
