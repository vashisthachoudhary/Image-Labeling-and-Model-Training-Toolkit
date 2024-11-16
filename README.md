# Image Labeling and Model Training Application

This project is an interactive Python-based GUI application for labeling images, training models, and testing predictions. It supports various machine learning algorithms like CNN, Random Forest, and Logistic Regression, with built-in hyperparameter tuning and visualization.

---

## Features

- **Image Labeling**: Annotate images using a paintbrush for binary segmentation.
- **Model Training**:
  - Supports **Convolutional Neural Networks (CNNs)** for pixel-wise prediction.
  - Includes **Random Forest** and **Logistic Regression** models with hyperparameter tuning.
  - Visualizations for hyperparameter tuning results.
- **Model Testing**: Test predictions on new images and view results in a separate window.
- **Customizable**: Easily extendable to add more models or preprocessing steps.
- **GUI Interface**: User-friendly interface built with `Tkinter`.

---

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/vashisthachoudhary/Image-Labeling-and-Model-Training-Toolkit.git
   cd Image-Labeling-and-Model-Training-Toolkit

3. **Install Required Dependencies**
   Install all necessary Python packages:
   ```bash
   pip install -r requirements.txt
   ```
   The requirements.txt file includes:
   ```bash
   numpy
   pandas
   scipy
   matplotlib
   scikit-learn
   tensorflow
   pillow
   joblib
   tk
   ```

## Usage
### Step 1: Label Images
1. Open the application.
2. Load images and use the paintbrush tool to annotate objects.
3. Save labeled images for training purposes.
### Step 2: Train a Model
1. Select a training algorithm:
- CNN: For deep learning-based segmentation.
- Random Forest: Machine learning with hyperparameter tuning.
- Logistic Regression: Simple classification-based approach.
2. Configure any desired hyperparameters.
3. Train the model and review the performance metrics.
### Step 3: Test a Model
1. Load a new image into the application.
2. un predictions and view the results with labeled outputs.
