from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the H5 model
MODEL_PATH = "model/maize-weedClassifier.h5"  
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image):
    """Preprocess image to match model input shape"""
    image = image.resize((256, 256))  # Change this if model expects 256x256
    image = np.array(image) / 255.0  # Normalize
    image = np.reshape(image, (1, 256, 256, 3))  # Match model shape
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    image = preprocess_image(image)
    
    # Save the image for display
    # Save the image for display
    filename = "uploaded_image.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.seek(0)  # Reset file pointer to save correctly
    file.save(filepath)  # Use Flask's file saving method
    
    # Run prediction
    prediction = model.predict(image)[0][0]  # Extract single prediction
    confidence = float(prediction) * 100  # Convert to percentage
    label = "Weed" if confidence > 50 else "Maize"  # Classify

    response = {"prediction": label, "image_url": filename}
    if label == "Weed":
        response["confidence"] = f"{confidence:.2f}%"

    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
