import tensorflow as tf
import numpy as np
import cv2

# Load the trained model and compile it
model = tf.keras.models.load_model("apple_banana_classifier.h5", compile=False)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])  # Recompile

# Define class labels
class_labels = ["Apple", "Banana"]

# Load and preprocess the test image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (150, 150))  # Resize to match training size
    img = img.astype("float32") / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Test on a single image
# image_path = "dataset/validation/banana/banana10.jpeg"  # Change to your test image path
image_path = "dataset/validation/apple/apple_logo.jpeg"  # Change to your test image path
image = preprocess_image(image_path)
prediction = model.predict(image)

# Print the result (Predicted class)
predicted_class = class_labels[int(prediction[0][0] > 0.5)]  # Using prediction[0][0] to avoid the deprecated operation
confidence = prediction[0][0] * 100 if predicted_class == "Banana" else (1 - prediction[0][0]) * 100

# Get the model's predicted probabilities for both classes
predicted_probabilities = model.predict(image)[0]  # Get probabilities for both classes
apple_prob = 1 - predicted_probabilities  # Probability for "Apple"
banana_prob = predicted_probabilities  # Probability for "Banana"

# Get the weights of the final Dense layer (used for prediction)
final_layer_weights = model.layers[-1].get_weights()[0]  # Weights of the last Dense layer
biases = model.layers[-1].get_weights()[1]  # Biases of the last Dense layer

print(f"Predicted class: {predicted_class} with {confidence:.2f}% confidence")
print(f"Apple Probability: {apple_prob[0] * 100:.2f}%")
print(f"Banana Probability: {banana_prob[0] * 100:.2f}%")

# Displaying additional model weights information
print("\nModel weights for the final dense layer:")
print(f"Final Layer Weights (size): {final_layer_weights.shape}")
print(f"Final Layer Biases: {biases}")

# If you want to see the actual raw prediction score (logits), you can access them too:
raw_prediction_score = model.layers[-1].output
print(f"Raw prediction score (logits): {raw_prediction_score}")
