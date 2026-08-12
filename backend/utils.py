import io
import base64
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import cv2
from fpdf import FPDF
import tempfile
import os


# ─────────────────────────────────────────────
# Image Preprocessing
# ─────────────────────────────────────────────

IMG_SIZE = (224, 224)


def preprocess_image(image_bytes: bytes) -> tuple[np.ndarray, Image.Image]:
    """
    Preprocess raw image bytes for model inference.
    Uses EfficientNet's native preprocess_input for correct normalization.

    Returns:
        img_array: numpy array of shape (1, 224, 224, 3) preprocessed for EfficientNet
        original_image: PIL Image for overlay visualization
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_image = image.copy()

    image = image.resize(IMG_SIZE)
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    return img_array, original_image


# ─────────────────────────────────────────────
# Grad-CAM Visualization
# ─────────────────────────────────────────────

def generate_gradcam(model, img_array: np.ndarray) -> np.ndarray:
    """
    Generate Grad-CAM heatmap for the given image.
    Compatible with Keras 3 Sequential models containing EfficientNet.

    Args:
        model: Trained Keras Sequential model (EfficientNet + head layers)
        img_array: Preprocessed image array of shape (1, 224, 224, 3)

    Returns:
        heatmap: numpy array of shape (224, 224) with values in [0, 1]
    """
    # The Sequential model has: [EfficientNet, GlobalAvgPool, Dense, Dropout, Dense]
    # EfficientNet is a Functional model, so .input/.output work on it.
    base_model = model.layers[0]  # EfficientNet (Functional model)
    head_layers = model.layers[1:]  # GAP, Dense, Dropout, Dense

    # Find the last conv layer inside EfficientNet
    last_conv_layer = None
    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        raise ValueError("No Conv2D layer found in base model.")

    # Build a sub-model from EfficientNet that outputs both
    # the conv layer activations and the final base output
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[last_conv_layer.output, base_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, base_output = grad_model(img_array)
        # Pass base_model output through the head layers
        x = base_output
        for layer in head_layers:
            x = layer(x, training=False)
        loss = x[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the feature maps by the pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU and normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Resize to image dimensions
    heatmap = cv2.resize(heatmap, IMG_SIZE)

    return heatmap


def overlay_heatmap(heatmap: np.ndarray, original_image: Image.Image, alpha: float = 0.4) -> Image.Image:
    """
    Overlay Grad-CAM heatmap on the original image.

    Args:
        heatmap: numpy array of shape (H, W) with values in [0, 1]
        original_image: PIL Image (original X-ray)
        alpha: blending factor for the heatmap overlay

    Returns:
        PIL Image with heatmap overlay
    """
    # Resize original to match heatmap
    original_resized = original_image.resize(IMG_SIZE)
    original_array = np.array(original_resized)

    # Convert heatmap to colored version (JET colormap)
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend
    overlay = np.uint8(alpha * heatmap_colored + (1 - alpha) * original_array)
    return Image.fromarray(overlay)


# ─────────────────────────────────────────────
# Report Generation
# ─────────────────────────────────────────────

def generate_report(prediction: str, confidence: float) -> str:
    """
    Generate a professional clinical report summary.
    """
    confidence_pct = f"{confidence * 100:.2f}%"
    
    report_lines = [
        "AURORA CLINICAL DIAGNOSTIC LOG",
        "──────────────────────────────",
        f"DETERMINATION : {prediction}",
        f"PROBABILITY   : {confidence_pct}",
        "METHODOLOGY   : EfficientNet-B0 + Grad-CAM",
        "──────────────────────────────",
        "\n[CLINICAL FINDINGS]"
    ]

    if prediction == "ABNORMAL":
        report_lines += [
            "• Neural engine identifies high-probability structural",
            "  irregularities (fracture/dislocation/lesion).",
            "• Visual activation hotspots indicate significant",
            "  morphological variance in the shoulder girdle.",
            "\n[URGENT RECOMMENDATIONS]",
            "1. Priority review by attending Radiologist.",
            "2. Orthopedic consultation for structural stabilization.",
            "3. Correlation with trauma history and physical exams."
        ]
    else:
        report_lines += [
            "• Radiological features within expected normative ranges.",
            "• No high-confidence structural abnormalities detected.",
            "\n[ROUTINE RECOMMENDATIONS]",
            "1. Routine clinical follow-up as symptomatic.",
            "2. MRI for soft-tissue assessment if symptoms persist.",
            "3. Final clearance subject to physician over-read."
        ]

    return "\n".join(report_lines)


def generate_pdf_report(
    prediction: str,
    confidence: float,
    original_image: Image.Image,
    heatmap_image: Image.Image,
    report_text: str
) -> bytes:
    """
    Generate a high-fidelity, professional Aurora Clinical Diagnostic Report.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # ── Aurora Professional Header ──
    # Top bar accent
    pdf.set_fill_color(16, 185, 129) # Emerald Primary
    pdf.rect(0, 0, 210, 3, 'F')
    
    # Header Content
    pdf.set_y(15)
    pdf.set_x(20)
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(15, 23, 42) # Deep Navy
    pdf.cell(0, 10, "SHOULDER.AI", ln=True)
    
    pdf.set_x(20)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(100, 116, 139) # Slate
    pdf.cell(0, 5, "AURORA CLINICAL EVOLUTION | NEURAL DIAGNOSTICS", ln=True)
    
    # Metadata Right Aligned
    import time
    patient_id = f"REF-{int(time.time() % 1000000):06d}"
    study_date = time.strftime("%d %b %Y | %H:%M:%S")
    
    pdf.set_y(15)
    pdf.set_x(-80)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(60, 5, f"CASE ID: {patient_id}", align='R', ln=True)
    pdf.set_x(-80)
    pdf.cell(60, 5, f"DATE: {study_date}", align='R', ln=True)
    pdf.set_x(-80)
    pdf.cell(60, 5, "STATUS: OFFICIAL REPRODUCTION", align='R', ln=True)
    
    pdf.ln(15)
    
    # ── Section 1: Diagnostic Determination (MAIN HIGHLIGHT) ──
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(15, 23, 42)
    pdf.set_x(20)
    pdf.cell(0, 10, "1. CLINICAL DETERMINATION", ln=True)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(5)
    
    # Result Box
    bg_color = (254, 242, 242) if prediction == "ABNORMAL" else (240, 253, 244)
    text_color = (153, 27, 27) if prediction == "ABNORMAL" else (21, 128, 61)
    accent_color = (244, 63, 94) if prediction == "ABNORMAL" else (16, 185, 129)
    
    pdf.set_fill_color(*bg_color)
    pdf.set_draw_color(*accent_color)
    pdf.set_line_width(0.8)
    
    current_y = pdf.get_y()
    pdf.rect(20, current_y, 170, 30, 'FD')
    
    pdf.set_y(current_y + 8)
    pdf.set_x(30)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*text_color)
    pdf.cell(0, 7, f"AI PREDICTION: {prediction}", ln=True)
    
    pdf.set_x(30)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(0, 7, f"Diagnostic Confidence: {confidence*100:.2f}% (High-Precision Mapping)", ln=True)
    
    pdf.ln(18)
    
    # ── Section 2: Visual Evidence (THE 'WHY') ──
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(15, 23, 42)
    pdf.set_x(20)
    pdf.cell(0, 10, "2. NEURAL FEATURE MAPPING (Grad-CAM)", ln=True)
    pdf.ln(2)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        orig_path = os.path.join(tmp_dir, "orig.png")
        heat_path = os.path.join(tmp_dir, "heat.png")
        original_image.convert("RGB").save(orig_path)
        heatmap_image.convert("RGB").save(heat_path)
        
        img_y = pdf.get_y()
        # Side by side with spacing
        pdf.image(orig_path, x=22, y=img_y, w=82)
        pdf.image(heat_path, x=106, y=img_y, w=82)
        
        pdf.set_y(img_y + 85)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(148, 163, 184)
        pdf.set_x(22)
        pdf.cell(82, 5, "EXAMINEE RADIOGRAPH", align='C')
        pdf.set_x(106)
        pdf.cell(82, 5, "NEURAL ATTENTION TARGETS", align='C', ln=True)
    
    pdf.ln(8)
    
    # ── Section 3: Clinical Insights & Log ──
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(15, 23, 42)
    pdf.set_x(20)
    pdf.cell(0, 10, "3. RADIOLOGICAL INSIGHTS", ln=True)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(5)
    
    pdf.set_font("Courier", "", 10)
    pdf.set_text_color(15, 23, 42)
    clean_report = report_text.encode("ascii", "ignore").decode("ascii")
    for line in clean_report.split('\n'):
        if line.strip():
            pdf.set_x(25)
            pdf.multi_cell(160, 6, line.strip())
            
    # ── Footer — Professional Disclaimer ──
    pdf.set_y(-35)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(148, 163, 184)
    disclaimer = "This report was generated via deep-learning feature extraction. It constitutes an automated radiological preliminary review. Findings must be validated by a licensed clinical radiologist prior to any surgical or therapeutic intervention. Aurora Diagnostics v4.1"
    pdf.set_x(20)
    pdf.multi_cell(170, 4, disclaimer, align='C')
    
    # Dynamic page number
    pdf.set_y(-15)
    pdf.set_font("Helvetica", "", 8)
    pdf.cell(0, 10, f"Page {pdf.page_no()}", align='C')
    
    return bytes(pdf.output())


# ─────────────────────────────────────────────
# Image Encoding
# ─────────────────────────────────────────────

def image_to_base64(image: Image.Image) -> str:
    """Convert a PIL Image to a base64-encoded PNG string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")
