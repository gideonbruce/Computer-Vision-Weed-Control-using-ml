import tensorflow as tf
import numpy as np
from PIL import Image

# Load your model
model = tf.keras.models.load_model("model/maize-weedClassifier.h5")

# Load and preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Ensure 3 channels
    image = image.resize((256, 256))  # Resize to match model input
    image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array  # Shape: (1, 256, 256, 3)

# Run the test
image_path = "maize3.jpg"
image = preprocess_image(image_path)

# Debug: Print shape
print("Processed image shape:", image.shape)  # Should be (1, 256, 256, 3)

# Predict
prediction = model.predict(image)

# Print results
#print("Prediction:", prediction)
print("Raw Prediction:", prediction)
print("Predicted Class:", "Weed" if prediction[0][0] > 0.5 else "Maize")
confidence = prediction[0][0]  
print(f"Predicted Class: {'Weed' if confidence > 0.5 else 'Maize'} (Confidence: {confidence:.2f})")



