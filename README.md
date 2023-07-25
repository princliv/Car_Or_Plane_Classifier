# Image Classifier GUI using Keras and Tkinter

![Coding](https://cdn-icons-png.flaticon.com/512/7580/7580978.png)

This is a Python program that implements an Image Classifier using a pre-trained Convolutional Neural Network (CNN) model with Keras and provides a Graphical User Interface (GUI) using Tkinter. The program allows users to classify images as either "plane" or "car." The GUI allows users to select an image file from their local storage, and the program will display the image along with the classification result and its probability.

## Table of Contents
1. [Introduction](#introduction)
2. [Model Architecture](#model-architecture)
3. [Data Generators](#data-generators)
4. [Training the Model](#training-the-model)
5. [GUI Description](#gui-description)
6. [How to Use the GUI](#how-to-use-the-gui)
7. [Installation](#installation)
8. [Improvements and Future Enhancements](#improvements-and-future-enhancements)

## Introduction
This program uses a pre-trained CNN model to classify images as either "plane" or "car." The model is implemented using Keras and is trained on image data for 20 epochs. The GUI is created using Tkinter and allows users to browse and classify their own images.

## Model Architecture
The CNN model consists of several layers:
1. Convolutional layers with ReLU activation and MaxPooling for feature extraction.
2. A Flatten layer to flatten the output of the convolutional layers.
3. Dense (fully connected) layers with ReLU activation and Dropout for classification.
4. The final Dense layer with a sigmoid activation for binary classification.

## Data Generators
The program uses the Keras ImageDataGenerator to load and preprocess the image data. The images are read from directories, and data augmentation is performed by rescaling the pixel values to a range between 0 and 1.

## Training the Model
The model is trained using the `model.fit` method with the training data generator. The training is performed for 20 epochs.

## GUI Description
The GUI has the following components:
- A title label displaying "IMAGE CLASSIFIER."
- A button labeled "Browse and Classify Image" that allows users to browse and select an image file for classification.
- A label displaying a default message "Click on the button given below."
- A label that displays the selected image and the predicted classification (plane or car) with its probability.
- The background of the GUI is set to an image displayed using the ImageTk library.

## How to Use the GUI
1. Clone or download this repository to your local machine.
2. Install the required libraries by running the following command in your terminal or command prompt:
	```bash
	python image_classifier_gui.py
	```
3. Place the "icon.png," "background.jpg," and the "v_data" folder (containing the training and testing data) in the same directory as the Python script.
4. Run the Python script using the following command:
	```bash
	pip install numpy matplotlib keras pillow
	```
5. The GUI window will open, displaying the title and a "Browse and Classify Image" button.
6. Click on the "Browse and Classify Image" button to open a file dialog. Select an image file (JPG, PNG, or JPEG).
7. The GUI will display the selected image and the predicted classification (plane or car) with its probability.

## Improvements and Future Enhancements
- Instead of using a pre-trained model, users can be allowed to train their own model using their custom dataset.
- Add support for classifying more categories of images, not just planes and cars.
- Improve the GUI layout and design to make it more appealing and user-friendly.
- Implement error handling to deal with cases where the user selects an invalid or unsupported file type.
- Add buttons or functionality to display the accuracy and loss graphs after training the model.
- Allow users to select and use different pre-trained models for image classification.
- Provide an option to save the classification results for future reference or analysis.

Feel free to use and modify this README file according to your needs. Happy coding!
