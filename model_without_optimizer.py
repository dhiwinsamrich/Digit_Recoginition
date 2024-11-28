import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import tkinter as tk


# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Define the model
model = models.Sequential([
    layers.Dense(
        32, 
        activation='relu', 
        input_shape=(28 * 28,), 
        kernel_regularizer=regularizers.l2(0.002)),
    layers.Dense(
        32, 
        activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(
        10, 
        activation='softmax')
])

model.compile(
    loss='categorical_crossentropy', 
    metrics=['accuracy'])

# Train the model
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(train_images, train_labels, epochs=20, batch_size=40, validation_split=0.2, callbacks=[early_stopping])

# Save the model after training
model.save("model_without_optimizer.h5")
print("Model saved successfully.")

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc:.4f}')