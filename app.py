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
    image = image.resize((256, 256))  # Resize if model expects 256x256
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.reshape(image, (1, 256, 256, 3))  # Ensure correct shape
    return image

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    image = preprocess_image(image)

    print("Expected model input shape:", model.input_shape)

    # Run prediction
    prediction = model.predict(image)[0][0]  # Extract single prediction
    confidence = float(prediction) * 100  # Convert to percentage
    label = "Weed" if prediction > 0.5 else "Maize"  # Classify

    # Return confidence only for "Weed"
    response = {"prediction": label}
    if label == "Weed":
        response["confidence"] = f"{confidence:.2f}%"

    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
