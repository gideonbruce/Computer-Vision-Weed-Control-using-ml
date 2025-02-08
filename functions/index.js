const functions = require("firebase-functions");
const express = require("express");
const cors = require("cors");
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");

const app = express();
app.use(cors({ origin: true }));

// Load the model
let model;
async function loadModel() {
    model = await tf.loadLayersModel("file://model/maize-weedClassifier.json");
    console.log("Model loaded successfully!");
}
loadModel();

// API to handle image classification
app.post("/predict", async (req, res) => {
    try {
        const imageBuffer = req.body.image;
        const tensor = tf.node.decodeImage(imageBuffer)
            .resizeNearestNeighbor([256, 256])
            .expandDims()
            .toFloat()
            .div(tf.scalar(255.0));

        const prediction = model.predict(tensor).dataSync();
        const confidence = (prediction[0] * 100).toFixed(2);
        const label = prediction[0] > 0.5 ? "Weed" : "Maize";

        res.json({ label, confidence: `${confidence}%` });
    } catch (error) {
        res.status(500).send(error.message);
    }
});

// Export the function
exports.api = functions.https.onRequest(app);
