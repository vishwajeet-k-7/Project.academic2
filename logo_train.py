import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam

# Set the path to the dataset
dataset_path = 'logo_dataset/'

# Set the image size for resizing
image_size = (128, 128)

def load_dataset():
    images = []
    labels = []

    # Load genuine document images
    genuine_path = os.path.join(dataset_path, 'Original_logo')
    for filename in os.listdir(genuine_path):
        file_path = os.path.join(genuine_path, filename)

        # Check if the file is an image (you can add more image formats as needed)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img = cv2.imread(file_path)

            if img is not None:
                img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
                images.append(img)
                labels.append(0)  # Genuine label = 0
            else:
                print(f"Error reading image: {file_path}")

    # Load forged document images
    forged_path = os.path.join(dataset_path, 'Forged_logo')
    for filename in os.listdir(forged_path):
        file_path = os.path.join(forged_path, filename)

        # Check if the file is an image (you can add more image formats as needed)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img = cv2.imread(file_path)

            if img is not None:
                img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
                images.append(img)
                labels.append(1)  # Forged label = 1
            else:
                print(f"Error reading image: {file_path}")

    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    return images, labels

def build_model(input_shape):
    model = Sequential()

    # Add convolutional layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    # Add dense layers
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model

def train_and_save_model():
    # Load the dataset
    images, labels = load_dataset()

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Normalize the pixel values to the range [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Build the model
    input_shape = X_train[0].shape
    model = build_model(input_shape)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test))

    # Save the trained model to an h5 file
    model.save('logo_model.h5')
    print("Trained model has been saved to 'logo_model.h5'.")

if __name__ == '__main__':
    train_and_save_model()
