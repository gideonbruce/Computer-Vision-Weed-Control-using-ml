from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the H5 model
MODEL_PATH = "model/maize-weedClassifier.h5"  
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image):
    """Preprocess image to match model input shape"""
    image = image.resize((224, 224))  # Resize to match model input
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    image = preprocess_image(image)

    # Run prediction
    prediction = model.predict(image)[0][0]  # Extract single prediction
    confidence = float(prediction) * 100  # Convert to percentage
    label = "Weed" if confidence > 50 else "Maize"  # Classify

    return jsonify({"prediction": label, "confidence": f"{confidence:.2f}%"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
