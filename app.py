# app.py
from flask import Flask, request, jsonify
from werkzeug.exceptions import RequestEntityTooLarge
from PIL import Image
import io

from model_wrapper import AssetClassifier

MODEL_PATH = "model_asset_classifier.h5"
classifier = AssetClassifier(MODEL_PATH)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 6 * 1024 * 1024  # 6 MB limit (accounts for base64/multipart overhead)

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({
        "error": "Uploaded file is too large",
        "limit_bytes": app.config["MAX_CONTENT_LENGTH"],
        "received_bytes": request.content_length
    }), 413

@app.route("/")
def index():
    return "Asset corrosion classifier – API is live."

@app.route("/classify/", methods=["POST"])
def classify():
    if "file" not in request.files:
        return jsonify({"error": "No file part named 'file'"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img_bytes = file.read()
        pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Cannot decode image: {e}"}), 400

    result = classifier.predict(pil_img)
    return jsonify(result)

if __name__ == "__main__":
    # For production use a WSGI server (gunicorn, waitress, etc.)
    app.run(host="0.0.0.0", port=5000, debug=False)