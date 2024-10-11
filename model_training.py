import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix
import numpy as np

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize data
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Create the model
model = models.Sequential()

# 6-layer model
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='nadam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), verbose=1)

# Predictions
y_pred = np.argmax(model.predict(x_test), axis=-1)
y_test_labels = np.argmax(y_test, axis=-1)

# Confusion matrix
conf_mat = confusion_matrix(y_test_labels, y_pred)
print('\nConfusion Matrix:')
print(conf_mat)

# Calculate Accuracy, Precision, Recall, F1-Score, and Specificity
accuracy = np.sum(np.diag(conf_mat)) / np.sum(conf_mat)
precision = np.diag(conf_mat) / np.sum(conf_mat, axis=0)
recall = np.diag(conf_mat) / np.sum(conf_mat, axis=1)
f1_score = 2 * (precision * recall) / (precision + recall)
specificity = []
for i in range(len(conf_mat)):
    true_negatives = np.sum(np.delete(np.delete(conf_mat, i, axis=0), i, axis=1))
    false_positives = np.sum(conf_mat[:, i]) - conf_mat[i, i]
    specificity.append(true_negatives / (true_negatives + false_positives))
specificity = np.array(specificity)

# Calculate the averages
avg_precision = np.mean(precision)
avg_recall = np.mean(recall)
avg_f1_score = np.mean(f1_score)
avg_specificity = np.mean(specificity)

# Print the results and averages
print('\nAccuracy:', accuracy)
print('Average Precision:', "{:.3f}".format(avg_precision))
print('Average Recall:', "{:.3f}".format(avg_recall))
print('Average F1-Score:', "{:.3f}".format(avg_f1_score))
print('Average Specificity:', "{:.3f}".format(avg_specificity))
