import os
import cv2
import numpy as np
from keras.utils import to_categorical, pad_sequences
from keras.layers import Dense, LSTM, Conv2D, MaxPooling2D, BatchNormalization, Bidirectional, Flatten, RepeatVector
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Define the data path
data_path = './data'

# Load the dataset
images = []
labels = []

# Read labels and find path to images from the words.txt file
with open(os.path.join(data_path, 'ascii', 'words.txt'), 'r') as file:
    for line in file:
        if not line.startswith('#'):
            line_parts = line.strip().split(' ')
            image_filename = line_parts[0] + '.png'
            image_subfolder1 = line_parts[0].split('-')[0]
            image_subfolder2 = line_parts[0].split('-')
            image_subfolder2 = image_subfolder2[0] + '-' + image_subfolder2[1]
            image_path = os.path.join(data_path, 'words', image_subfolder1, image_subfolder2, image_filename)
            label = line_parts[-1]
            images.append(image_path)
            labels.append(label)

print('Number of images and corresponding labels: ', len(images))

# Define lists to hold images and labels
processed_images = []
processed_labels = []

# Reading images from image_path and processing
for image_path, label in zip(images, labels):
    # Read images in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is not None:
        # Remove salt and pepper
        image = cv2.medianBlur(image, 5)
        # resize images
        image = cv2.resize(image, (128, 32))
        # Normalizing images
        image = image.astype(np.float32) / 255.0
        processed_images.append(image)
        processed_labels.append(label)

# Create set of unique numerics to represent letters
char_set = set()
for label in processed_labels:
    char_set.update(label)
char_set = sorted(list(char_set))
num_classes = len(char_set)

char_to_idx = {char: idx for idx, char in enumerate(char_set)}
idx_to_char = {idx: char for idx, char in enumerate(char_set)}

# Encode the actual labels to numbers to use in the cnn model
encoded_labels = []
for label in processed_labels:
    encoded_label = [char_to_idx[char] for char in label if char in char_to_idx]
    encoded_labels.append(encoded_label)

# Pad labels to all be the same length
padded_labels = pad_sequences(encoded_labels, padding='post')

# Add an extra dimension to the images
processed_images = np.expand_dims(processed_images, axis=-1)

# Split the dataset into training, validation and test sets
X_train, X_val, y_train, y_val = train_test_split(processed_images, padded_labels, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

# Convert labels to binary matrix's, each row corresponds to a position in the sequence. while the columns
# corresponds to a character class
y_train = to_categorical(y_train, num_classes=num_classes + 1)
y_val = to_categorical(y_val, num_classes=num_classes + 1)
y_test = to_categorical(y_test, num_classes=num_classes + 1)


# Define a learning rate schedule function
def schedule(epoch, learning_rate):
    if epoch < 5:
        return learning_rate  # Keep the initial learning rate for the first 5 epochs
    else:
        return learning_rate * 0.1  # Reduce the learning rate by a factor of 0.1 for subsequent epochs


# Create a learning rate scheduler callback
lr_scheduler = LearningRateScheduler(schedule)

# Define convolutinal neural network
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 128, 1)))  # Convolutional layer with 32 filters
model.add(BatchNormalization())  # Batch normalization layer
model.add(MaxPooling2D((2, 2)))  # Max pooling layer
model.add(Conv2D(64, (3, 3), activation='relu'))  # Convolutional layer with 64 filters
model.add(BatchNormalization())  # Batch normalization
model.add(MaxPooling2D((2, 2)))  # Max pooling layer
model.add(Conv2D(128, (3, 3), activation='relu'))  # convolutional layer with 128 filters
model.add(BatchNormalization())  # batch normalization layer
model.add(MaxPooling2D((2, 2)))  # Max pooling layer
model.add(Flatten())  # layer used to flatten into one dimensional vector
model.add(RepeatVector(padded_labels.shape[1]))  # repeats flattened vector equal to the length of the padded labels
model.add(Bidirectional(LSTM(256, return_sequences=True)))  # Bidirectional lstm layer with 256 units
model.add(Dense(num_classes + 1, activation='softmax'))  # probability distribution for possible characters

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

with tf.device('/GPU:0'):
    # Train the model
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              batch_size=16, epochs=10, callbacks=[lr_scheduler, EarlyStopping(patience=3)])

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)

# Print results of evaluation on the test set
print('Test loss:', loss)
print('Test accuracy:', accuracy)

model.save('my_model')
