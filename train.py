# Train CNN on MNIST Dataset for Handwritten Digit Recognition

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models, datasets

# Set fixed random seed for repeatable results
np.random.seed(42)
tf.random.set_seed(42)

# Task 1: Load the MNIST dataset from keras.datasets
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# Task 2: Visualize 10 sample digit images with labels and save
plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(str(y_train[i]))
    plt.axis('off')
plt.savefig('outputs/sample_digits.png')
plt.close()

# Task 3: Normalize pixel values to the range 0-1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Task 4: Reshape images to (28, 28, 1) for CNN input
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Task 5: Design a small CNN architecture
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Task 6: Compile and train for 5 epochs with batch size 128
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Task 7: Validate during training with 10% data and plot training curves
history = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)

# Plot training vs validation accuracy and loss side by side
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('outputs/training_curves.png')
plt.close()

# Task 8: Evaluate on the test set and print test accuracy and loss
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print('Test Accuracy:', test_accuracy)
print('Test Loss:', test_loss)

# Task 9: Generate confusion matrix and classification report
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)

# Plot and save confusion matrix heatmap
cm = confusion_matrix(y_test, predicted_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix')
plt.savefig('outputs/confusion_matrix.png')
plt.close()

# Print and save classification report
report = classification_report(y_test, predicted_labels)
print(report)
report_file = open('outputs/classification_report.txt', 'w')
report_file.write(report)
report_file.close()

# Task 10: Display 10 test images with Predicted vs Actual labels
plt.figure(figsize=(12, 3))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    label_text = 'P:' + str(predicted_labels[i]) + ' A:' + str(y_test[i])
    plt.title(label_text)
    plt.axis('off')
plt.tight_layout()
plt.savefig('outputs/predictions.png')
plt.close()

# Save the trained model to model folder
model.save('model/digit_cnn.keras')
print('Model saved to model/digit_cnn.keras')

