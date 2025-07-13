# food_calorie_model.py

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# Paths
DATASET_PATH = "food-101/images"  # Adjust based on your extracted dataset location

# Parameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5

# Data Generator
train_datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)
train_gen = train_datagen.flow_from_directory(DATASET_PATH,
                                              target_size=(IMG_SIZE, IMG_SIZE),
                                              batch_size=BATCH_SIZE,
                                              class_mode='categorical',
                                              subset='training')
val_gen = train_datagen.flow_from_directory(DATASET_PATH,
                                            target_size=(IMG_SIZE, IMG_SIZE),
                                            batch_size=BATCH_SIZE,
                                            class_mode='categorical',
                                            subset='validation')

# Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
x = GlobalAveragePooling2D()(base_model.output)
output = Dense(train_gen.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# Freeze base model
for layer in base_model.layers:
    layer.trainable = False

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# Save model and class mapping
model.save("food_model.h5")
with open("class_indices.json", "w") as f:
    import json
    json.dump(train_gen.class_indices, f)
