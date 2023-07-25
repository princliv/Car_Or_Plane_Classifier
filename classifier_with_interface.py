import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image

# Define image dimensions
img_width, img_height = 224, 224
input_shape = (img_width, img_height, 3)

# Define the model
model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Data generators
train_data_dir = 'v_data/train'  # No need to specify the full path, as it is in the same directory
test_data_dir = 'v_data/test'    # No need to specify the full path, as it is in the same directory

datagen = ImageDataGenerator(rescale=1. / 255)

train_data_gen = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=20,
    class_mode='binary'
)

test_data_gen = datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=20,
    class_mode='binary'
)

# Train the model
epochs = 20
history = model.fit(
    train_data_gen,
    epochs=epochs,
    validation_data=test_data_gen
)

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Create the main application window
root = tk.Tk()
root.title("Image Classifier")
root.geometry("1000x800")
root.iconphoto(True, tk.PhotoImage(file="icon.png"))

# Disable maximizing the window
root.resizable(False, False)

# Load the background image
background_image = Image.open("background.jpg")
background_image = background_image.resize((1000, 800))
background_photo = ImageTk.PhotoImage(background_image)

# Create a label to display the background image
background_label = tk.Label(root, image=background_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Add a title label
title_label = tk.Label(root, text="IMAGE CLASSIFIER", font=("Arial", 15, "bold"), bg='#101018', fg='white')
title_label.pack(pady=20)

# Function to open a file dialog and get the image path
def browse_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        classify_and_update(file_path)

# Create a label to display the image and prediction
image_label = tk.Label(root, bg='#1c1f2e', padx=10, pady=10)
image_label.pack(pady=10)

# Function to update the image and prediction label
def update_image_label(image_path, prediction_text):
    img = Image.open(image_path)
    img.thumbnail((400, 400))
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img

    prediction_label.config(text=prediction_text)

# Create a label to display the prediction text
prediction_label = tk.Label(root, text="", font=("Arial", 16), bg='#1c1f2e', fg='skyblue')
prediction_label.pack(pady=10)

empty = tk.Label(root, text=" Click on the button given below ", font='Arial 10', bg="#1c1f2e", fg="yellow")
empty.pack(pady=0)

# Placeholder function for load_and_predict (replace this with your actual image classification code)
def load_and_predict(image_path):
    # Your image classification code goes here.
    # You should load the image, preprocess it, and pass it through your trained model.
    img = Image.open(image_path)
    img = img.resize((img_width, img_height))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)[0][0]  # This assumes your model returns a single probability value
    return prediction

# Function to classify and update the image label and prediction
def classify_and_update(image_path):
    prediction = load_and_predict(image_path)

    if prediction >= 0.5:
        prediction_text = "It's a plane!"
    else:
        prediction_text = "It's a car!"

    update_image_label(image_path, prediction_text + f" (Probability: {round(prediction, 4)})")

# Function to allow the user to input their own image
def browse_and_classify():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        classify_and_update(file_path)

# Button to browse and classify the user's own image
classify_button = tk.Button(root, text="Browse and Classify Image", font=("Arial", 14), command=browse_and_classify,
                            bg="#ffc107", fg="black")
classify_button.pack(pady=20)

# Main loop to run the GUI application
root.mainloop()
