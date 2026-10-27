# Task 11: Predict handwritten digit from custom image

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

# Load the trained CNN model
model = tf.keras.models.load_model('model/digit_cnn.keras')

# Check for digit.png or image.png in my_digit folder
image_path = 'my_digit/digit.png'
if not os.path.exists(image_path):
    image_path = 'my_digit/image.png'

# Open image with Pillow, convert to grayscale, and resize to 28x28
img = Image.open(image_path)
img = img.convert('L')
img = img.resize((28, 28))

# Convert image to numpy array
img_array = np.array(img, dtype='float32')

# If background is white (mean > 127), invert to make digit white on black
if np.mean(img_array) > 127:
    img_array = 255.0 - img_array

# Normalize pixel values to range 0-1
img_array = img_array / 255.0

# Reshape image to (1, 28, 28, 1) for CNN input
img_input = img_array.reshape(1, 28, 28, 1)

# Predict digit probabilities
prediction = model.predict(img_input)
predicted_digit = int(np.argmax(prediction))
confidence = float(np.max(prediction)) * 100.0

# Print the predicted digit and confidence percentage
print('Predicted Digit:', predicted_digit)
print('Confidence:', round(confidence, 2), '%')

# Show image with prediction as title and save to outputs
plt.imshow(img_array, cmap='gray')
title_text = 'Predicted: ' + str(predicted_digit) + ' (' + str(round(confidence, 2)) + '%)'
plt.title(title_text)
plt.axis('off')
plt.savefig('outputs/my_digit_prediction.png')
plt.close()
