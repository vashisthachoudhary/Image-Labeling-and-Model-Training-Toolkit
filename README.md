﻿# Image-Labeling-and-Model-Training-Toolkit

Image Labeling and Model Training App
This Python application facilitates labeling images and training machine learning models. The tool is designed to enable users to preprocess image data, label images using pre-saved masks, train models, and evaluate predictions using different machine learning algorithms, including convolutional neural networks (CNNs), logistic regression, and random forest classifiers.

Features
Image Preprocessing: Automatically preprocesses images for training.
Label Management: Supports labeling images using pre-saved masks and converts foreground/background labels to binary format.
Model Training:
CNN: Trains convolutional neural networks for pixel-wise classification.
Random Forest: Trains and fine-tunes a random forest classifier using hyperparameter search.
Logistic Regression: Implements logistic regression with hyperparameter optimization.
Hyperparameter Optimization: Includes randomized search for finding the best hyperparameters for models.
Visualization: Generates plots to analyze model performance with different hyperparameters.
Model Testing: Predicts labels for new test images and visualizes results.
Save and Load Models: Saves trained models for future use and loads pre-trained models for testing.
Prerequisites
Before running this application, ensure you have the following:

Software Requirements
Python 3.8 or higher
Supported OS: Windows/Linux/MacOS
Libraries
Install required libraries using:

bash
Copy code
pip install -r requirements.txt
Required Libraries
tensorflow
numpy
pandas
matplotlib
scikit-learn
opencv-python
pillow
joblib
seaborn
scipy
