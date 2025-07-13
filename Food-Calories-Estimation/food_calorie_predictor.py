# food_calorie_predictor.py

import json
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tensorflow as tf
# Load model and class indices
model = load_model("food_model.h5")
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
    class_names = {v: k for k, v in class_indices.items()}

with open("calorie_mapping.json", "r") as f:
    calorie_map = json.load(f)

# Predict function
def predict_food(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img) / 255.
    x = np.expand_dims(x, axis=0)
    
    preds = model.predict(x)
    class_id = np.argmax(preds)
    class_name = class_names[class_id]
    calories = calorie_map.get(class_name, "Unknown")

    print(f"Food: {class_name.replace('_', ' ').title()}")
    print(f"Estimated Calories: {calories} kcal")

# Example usage
if __name__ == "__main__":
    img_path = "example.jpg"  # Change to your test image path
    predict_food(img_path)
