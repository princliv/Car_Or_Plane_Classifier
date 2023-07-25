<img align="right" alt="Coding" width="400" src="https://cdn-icons-png.flaticon.com/512/7580/7580978.png">

# **Image Classifier GUI using Keras and Tkinter**

This is a Python program that implements an Image Classifier using a pre-trained Convolutional Neural Network (CNN) model with Keras and provides a Graphical User Interface (GUI) using Tkinter. The program allows users to classify images as either "plane" or "car." The GUI allows users to select an image file from their local storage, and the program will display the image along with the classification result and its probability.

## Libraries Used:
- numpy: A library for numerical operations in Python.
- matplotlib.pyplot: Used for data visualization, particularly for plotting accuracy and loss graphs.
- keras: A deep learning library that provides a high-level neural networks API.
- tkinter: The standard Python interface to the Tk GUI toolkit, used for creating the GUI.
- PIL (Python Imaging Library): Used for image processing tasks.

## Model Architecture:
The CNN model used for classification consists of several layers:
1. Convolutional layers with ReLU activation and MaxPooling for feature extraction.
2. A Flatten layer to flatten the output of the convolutional layers.
3. Dense (fully connected) layers with ReLU activation and Dropout for classification.
4. The final Dense layer with a sigmoid activation for binary classification.

## Data Generators:
The program uses the Keras ImageDataGenerator to load and preprocess the image data. The images are read from directories,<br>
 and data augmentation is performed by rescaling the pixel values to a range between 0 and 1.

## Training the Model:
The model is trained using the `model.fit` method with the training data generator. The training is performed for 20 epochs.

## GUI Description:
The GUI has the following components:
- A title label displaying "IMAGE CLASSIFIER."
- A button labeled "Browse and Classify Image" that allows users to browse and select an image file for classification.
- A label displaying a default message "Click on the button given below."
- A label that displays the selected image and the predicted classification (plane or car) with its probability.
- The background of the GUI is set to an image displayed using the ImageTk library.

## How to Use the GUI:
1. Install the required libraries: numpy, matplotlib, Keras, tkinter, and PIL.
2. Place the "icon.png," "background.jpg," and the "v_data" folder 
	(containing the training and testing data) in the same directory as the Python script.
3. Run the Python script.
4. The GUI window will open, displaying the title and a "Browse and Classify Image" button.
5. Click on the "Browse and Classify Image" button to open a file dialog. Select an image file (JPG, PNG, or JPEG).
6. The GUI will display the selected image and the predicted classification (plane or car) with its probability.

## Note:
- The program assumes the user has a pre-trained model (which is already defined in the code). <br>
If you don't have a pre-trained model, you should first train a CNN model using image data related to planes and cars.
- The code provides a placeholder function `load_and_predict(image_path)` where you should replace it with your actual image classification code using the trained model.

## Improvements and Future Enhancements:
- Instead of using a pre-trained model, users can be allowed to train their own model using their custom dataset.
- Add support for classifying more categories of images, not just planes and cars.
- Improve the GUI layout and design to make it more appealing and user-friendly.
- Implement error handling to deal with cases where the user selects an invalid or unsupported file type.
- Add buttons or functionality to display the accuracy and loss graphs after training the model.
- Allow users to select and use different pre-trained models for image classification.
- Provide an option to save the classification results for future reference or analysis.
