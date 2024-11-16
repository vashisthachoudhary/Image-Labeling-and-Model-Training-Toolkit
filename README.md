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
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name

Install Dependencies Install the required Python packages using pip:

bash
Copy code
pip install -r requirements.txt
Example requirements:

plaintext
Copy code
numpy
pandas
matplotlib
scikit-learn
tensorflow
pillow
joblib
tk
Run the Application

bash
Copy code
python main.py
