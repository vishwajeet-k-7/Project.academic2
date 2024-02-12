import cv2
import numpy as np
from keras.models import load_model
import tkinter as tk
from tkinter import filedialog

def preprocess_image(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if img is None:
        print(f"Error: Unable to read the image from {image_path}")
        return None

    # Resize the image to the model's input shape
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

    # Normalize the pixel values to the range [0, 1]
    img = img.astype('float32') / 255.0

    # Expand the dimensions to match the model's input shape (add batch dimension)
    img = np.expand_dims(img, axis=0)

    return img

def predict_genuine_or_forged(image_path):
    # Load the trained model from the file
    model = load_model('signature_model.h5')

    # Preprocess the image
    img = preprocess_image(image_path)

    # Check if image preprocessing was successful
    if img is None:
        return None

    # Use the model to predict the label
    prediction = model.predict(img)
    print("prediction",prediction)

    # Get the predicted class (genuine or forged)
    if prediction[0][0] >= 0.5:
        result = 'forged'
    else:
        result = 'genuine'

    return result

def classify_image():
    image_path_to_classify = filedialog.askopenfilename(title="Select an Image File")
    if image_path_to_classify:
        result = predict_genuine_or_forged(image_path_to_classify)
        result_label.config(text=f'The document has {result} sign')

# Create a Tkinter window
root = tk.Tk()
root.title("Signature Classifier")

# Create a button to select an image
select_button = tk.Button(root, text="Select Image", command=classify_image)
select_button.pack(pady=10)

# Create a label to display the result
result_label = tk.Label(root, text="", font=("Helvetica", 16))
result_label.pack()

# Start the Tkinter event loop
root.mainloop()
