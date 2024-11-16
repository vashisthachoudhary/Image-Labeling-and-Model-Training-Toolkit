import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk, ImageDraw
import os
import numpy as np
import pandas as pd
# import pickle
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.stats import randint
from scipy.stats import loguniform
import matplotlib.pyplot as plt
class ImageLabelingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Labeling and Model Training Tool")
        
        # Initialize canvas and image variables
        self.canvas = tk.Canvas(root, width=500, height=500)
        self.canvas.pack()
        self.image = None
        self.draw = None
        self.label_image = None  # For storing the labeled image
        self.original_image = None  # Store original image
        self.current_image_index = 0
        self.image_list = []
        self.labels = []  # To store labels (painted masks)
        self.history = []  # To store history of label images for undo

        # Buttons for loading images and moving to the next image
        self.load_btn = tk.Button(root, text="Load Images", command=self.load_images)
        self.load_btn.pack(side=tk.LEFT)
        
        self.next_btn = tk.Button(root, text="Next Image", command=self.next_image)
        self.next_btn.pack(side=tk.LEFT)
        
        self.train_btn = tk.Button(root, text="Train Model", command=self.train_model)
        self.train_btn.pack(side=tk.LEFT)

        self.test_btn = tk.Button(root, text="Test Image", command=self.test_image)
        self.test_btn.pack(side=tk.LEFT)

        self.undo_btn = tk.Button(root, text="Undo", command=self.undo)
        self.undo_btn.pack(side=tk.LEFT)

        # Model selection dropdown with CNN added
        self.model_var = tk.StringVar()
        self.model_var.set("CNN")  # Default value
        self.model_dropdown = ttk.Combobox(root, textvariable=self.model_var, values=["Random Forest", "Logistic Regression", "CNN"])
        self.model_dropdown.pack(side=tk.LEFT)
        
        # Brush size selection
        self.brush_size_var = tk.IntVar(value=10)  # Default brush size
        self.brush_size_slider = tk.Scale(root, from_=1, to=50, orient=tk.HORIZONTAL, label="Brush Size", variable=self.brush_size_var)
        self.brush_size_slider.pack(side=tk.LEFT)

        # Set up drawing functionality
        self.canvas.bind("<B1-Motion>", self.paint)  # While dragging with the mouse pressed
        self.canvas.bind("<Button-1>", self.paint)    # On mouse click
        self.canvas.bind("<Button-3>", self.eraser)    # Right-click for eraser

    # (Other methods remain unchanged until `train_model`)
    def load_images(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.image_list = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('jpg', 'jpeg'))]
            self.labels = []  # Clear previous labels
            self.history = []  # Clear previous history
            self.current_image_index = 0  # Reset to first image
            self.display_image()

    def display_image(self):
        if self.image_list:
            image_path = self.image_list[self.current_image_index]
            self.original_image = Image.open(image_path).convert("RGBA")  # Convert to RGBA for alpha compositing
            self.image = self.original_image.copy()
            self.label_image = Image.new("RGBA", self.image.size, (255, 255, 255, 0))  # Transparent label image
            self.draw = ImageDraw.Draw(self.label_image)  # Prepare drawing on the label image
            self.history.append(self.label_image.copy())  # Save the initial label image

            self.update_canvas(self.image)

    def update_canvas(self, img):
        self.tk_image = ImageTk.PhotoImage(img.resize((500, 500)))  # Resize for display
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.image = self.tk_image  # Keep a reference to avoid garbage collection

    def paint(self, event):
        x, y = event.x, event.y
        
        # Get the current brush size
        brush_size = self.brush_size_var.get()

        # Adjust x, y for the original image size and the displayed canvas size
        canvas_width, canvas_height = 500, 500
        img_width, img_height = self.image.size
        scale_x = img_width / canvas_width
        scale_y = img_height / canvas_height
        x_original = int(x * scale_x)
        y_original = int(y * scale_y)

        # Draw an ellipse centered at the cursor location on the label image
        color = (255, 0, 0, 128)  # Semi-transparent red for the object
        self.draw.ellipse([x_original-brush_size, y_original-brush_size, x_original+brush_size, y_original+brush_size], fill=color, outline=color)

        # Update the displayed image
        combined_image = Image.alpha_composite(self.image, self.label_image)
        self.update_canvas(combined_image)

    def eraser(self, event):
        x, y = event.x, event.y

        # Transparent color for eraser
        color = (255, 255, 255, 0)  # Fully transparent for erasing

        # Get the current brush size
        brush_size = self.brush_size_var.get()

        # Adjust x, y for the original image size and the displayed canvas size
        canvas_width, canvas_height = 500, 500
        img_width, img_height = self.image.size
        scale_x = img_width / canvas_width
        scale_y = img_height / canvas_height
        x_original = int(x * scale_x)
        y_original = int(y * scale_y)

        # Draw an ellipse centered at the cursor location on the label image to erase
        self.draw.ellipse([x_original-brush_size, y_original-brush_size, x_original+brush_size, y_original+brush_size], fill=color)

        # Update the displayed image
        combined_image = Image.alpha_composite(self.image, self.label_image)
        self.update_canvas(combined_image)

    def undo(self):
        if self.history:
            self.history.pop()  # Remove the last image from history
            if self.history:
                self.label_image = self.history[-1].copy()  # Get the last image in history
            else:
                self.label_image = Image.new("RGBA", self.image.size, (255, 255, 255, 0))  # Reset to a new transparent image
            # Update the displayed image
            combined_image = Image.alpha_composite(self.image, self.label_image)
            self.update_canvas(combined_image)

    def next_image(self):
        if self.image_list and self.current_image_index < len(self.image_list) - 1:
            self.auto_save_label()  # Auto-save the current label
            self.current_image_index += 1
            self.display_image()

   
    def auto_save_label(self):
        if self.label_image:
            # Convert label image to grayscale for saving
            label_output = self.label_image.convert("L")  # Convert to grayscale
            label_array = np.array(label_output)  # Store the label as a numpy array

            # Resize to the same target size as the input images
            label_array_resized = np.array(label_output.resize((100, 100)))  # Resize for consistency
            self.labels.append(label_array_resized)  # Append the label for the image
            save_path = os.path.splitext(self.image_list[self.current_image_index])[0] + "_label.png"  # Save as PNG
            label_output.save(save_path)  # Save the label image as .png
            print(f"Auto-saved label image: {save_path}")


    def preprocess_image(self, image_path, target_size=(100, 100)):
        
        # Preprocess input image to resize and normalize.
        # Convert images to RGB format.
        
        img = Image.open(image_path).resize(target_size).convert("RGB")  # Convert to RGB
        img_array = np.array(img) / 255.0  # Normalize pixel values

        return img_array

    def pixelwise_data_prep(self, image_array):
        
        # Prepare pixel data for training: reshape and scale.
        # Ensure correct shape for RGB images.
        
        if image_array.ndim == 2:  # If grayscale
            # If for some reason, grayscale, convert it to RGB
            image_array = np.stack((image_array,)*3, axis=-1)

        pixel_data = image_array.reshape(-1, 3)  # Flatten the image array to (num_pixels, 3)
        return pixel_data  # Keep pixel values as is (already scaled)
    

    def augment_images_and_masks(image_paths, mask_paths, batch_size=1, target_size=(100, 100)):
        # Define ImageDataGenerator with augmentation parameters
        data_gen_args = dict(
            rotation_range=30,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='reflect'
        )
    
        image_datagen = ImageDataGenerator(**data_gen_args)
        mask_datagen = ImageDataGenerator(**data_gen_args)
        def load_image(path, target_size=(100, 100)):
            return np.array(Image.open(path).resize(target_size))

        def load_mask(path, target_size=(100, 100)):
            return np.array(Image.open(path).resize(target_size).convert('L'))
        
        # Load images and masks as arrays with fixed target size
        images = np.array([load_image(img_path, target_size) for img_path in image_paths])
        masks = np.array([load_mask(mask_path, target_size) for mask_path in mask_paths])
    
        # Reshape masks to add channel dimension for ImageDataGenerator
        masks = np.expand_dims(masks, axis=-1)
    
        # Initialize generators for images and masks
        image_generator = image_datagen.flow(images, batch_size=batch_size, seed=42)
        mask_generator = mask_datagen.flow(masks, batch_size=batch_size, seed=42)
    
        # Generate augmented images and masks
        augmented_images = []
        augmented_masks = []
        for img_batch, mask_batch in zip(image_generator, mask_generator):
            augmented_images.extend(img_batch.astype(np.uint8))
            augmented_masks.extend(mask_batch.astype(np.uint8).squeeze())
            
            if len(augmented_images) >= len(image_paths):  # Only generate one batch
                break
            
        return augmented_images[:len(image_paths)], augmented_masks[:len(mask_paths)]

    

    def train_model(self):
        model_choice = self.model_var.get()
        if model_choice == "CNN":
            X, y = [], []

            for img_path in self.image_list:
                img_array = self.preprocess_image(img_path)  # Preprocess image for training

                # Look for a saved mask corresponding to this image with .png extension
                mask_path = img_path.replace(".jpg", "_label.png")  # Assuming mask file naming pattern with .png extension
                if not os.path.exists(mask_path):
                    print(f"No saved mask found for {img_path}. Skipping.")
                    continue
                
                # Load the saved mask
                mask = Image.open(mask_path).convert("L")  # Convert to grayscale (L mode)


                mask_resized = mask.resize((100, 100), Image.NEAREST)
                mask_array = np.array(mask_resized)

                # Convert 255 to 1 (foreground) and 0 for background
                mask_array = np.where(mask_array == 255, 1, 0)

                X.append(img_array)
                y.append(mask_array)

            # Convert X and y to numpy arrays and apply categorical labels for CNN
            X = np.array(X)
            y = np.array(y)  # We do not apply to_categorical for pixel-wise masks, use binary encoding directly for training

            # Ensure there’s data to train with
            if len(X) == 0 or len(y) == 0:
                print("No training data available. Please ensure images and masks are present.")
                return

            # Split the dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Train CNN
            def create_cnn_model(learning_rate=0.001, filters=32, kernel_size=(3, 3), depth=0):
                model = Sequential()

                # Initial Conv and Pool layers
                model.add(Conv2D(filters, kernel_size=kernel_size, activation='relu', padding='same', input_shape=(100, 100, 3)))
                model.add(MaxPooling2D(pool_size=(2, 2)))

                # Adding Conv-Pool layers based on depth
                for i in range(1, depth):
                    model.add(Conv2D(filters * (2 ** i), kernel_size=kernel_size, activation='relu', padding='same'))

                    # Only add MaxPooling if dimensions are larger than 1x1
                    model.add(MaxPooling2D(pool_size=(2, 2)))

                    # Check if spatial dimension will reach 1x1, stop if true
                    if (100 // (2 ** (i+1))) < 2:
                        break
                    
                # UpSampling and Conv layers to restore image dimensions
                for i in range(depth, 1, -1):
                    model.add(Conv2D(filters * (2 ** i), kernel_size=kernel_size, activation='relu', padding='same'))
                    model.add(UpSampling2D(size=(2, 2)))

                model.add(UpSampling2D(size=(2, 2)))
                model.add(Conv2D(1, kernel_size=(1, 1), activation='sigmoid', padding='same'))  # Output layer with 1 channel

                # Compile the model
                model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
                return model
            acc = []
            temp = []
            for i in [0.01,0.001,0.0001]:
                for j in [16,32,64]:
                    for k in [(3,3),(5,5),(7,7)]:
                        for l in range(0,5):
                            model = create_cnn_model(learning_rate=i, filters=j, kernel_size=k)
                            print(f"Training CNN model with learning rate {i}, filters {j}, kernel size {k}...")
                            model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test))
                            accuracy = model.evaluate(X_test, y_test)
                            temp = [i, j, k, l, accuracy[1]] 
                            acc.append(temp)
            max_accuracy = 0
            best_model = None
            for i in range(len(acc)):
                if acc[i][4] > max_accuracy:
                    max_accuracy = acc[i][4]
                    best_model = acc[i]
            print("Best model found:")
            print(f"Learning Rate: {best_model[0]}, Filters: {best_model[1]}, Kernel Size: {best_model[2]}, Accuracy: {best_model[3]}, Depth: {best_model[4]}")
            model = create_cnn_model(learning_rate=best_model[0], filters=best_model[1], kernel_size=best_model[2], depth=best_model[3])
            # model = create_cnn_model(learning_rate=0.0011, filters=64, kernel_size=(5,5))
            model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
            print("Model training completed.")
            # Create a plot for each combination of hyperparameters (color-coded)
            learning_rates = [config[0] for config in acc]
            filter_sizes = [config[1] for config in acc]
            kernel_sizes = [config[2] for config in acc]
            depths = [config[3] for config in acc]
            accuracies = [config[4] for config in acc]

            # Plot 1: Learning Rate vs Accuracy
            plt.figure(figsize=(10, 6))
            for filter_size in set(filter_sizes):
                plt.plot(
                    [learning_rates[i] for i in range(len(acc)) if filter_sizes[i] == filter_size],
                    [accuracies[i] for i in range(len(acc)) if filter_sizes[i] == filter_size],
                    label=f'Filters: {filter_size}'
                )
            plt.xlabel('Learning Rate')
            plt.ylabel('Accuracy')
            plt.title('Learning Rate vs Accuracy')
            plt.legend()
            plt.grid(True)
            plt.show()

            # Plot 2: Filter Size vs Accuracy
            plt.figure(figsize=(10, 6))
            for lr in set(learning_rates):
                plt.plot(
                    [filter_sizes[i] for i in range(len(acc)) if learning_rates[i] == lr],
                    [accuracies[i] for i in range(len(acc)) if learning_rates[i] == lr],
                    label=f'Learning Rate: {lr}'
                )
            plt.xlabel('Filter Size')
            plt.ylabel('Accuracy')
            plt.title('Filter Size vs Accuracy')
            plt.legend()
            plt.grid(True)
            plt.show()

            # Plot 3: Kernel Size vs Accuracy
            plt.figure(figsize=(10, 6))
            for depth_val in set(depths):
                plt.plot(
                    [str(kernel_sizes[i]) for i in range(len(acc)) if depths[i] == depth_val],
                    [accuracies[i] for i in range(len(acc)) if depths[i] == depth_val],
                    label=f'Depth: {depth_val}'
                )
            plt.xlabel('Kernel Size')
            plt.ylabel('Accuracy')
            plt.title('Kernel Size vs Accuracy')
            plt.legend()
            plt.grid(True)
            plt.show()

            # Plot 4: Depth vs Accuracy
            plt.figure(figsize=(10, 6))
            for lr in set(learning_rates):
                plt.plot(
                    [depths[i] for i in range(len(acc)) if learning_rates[i] == lr],
                    [accuracies[i] for i in range(len(acc)) if learning_rates[i] == lr],
                    label=f'Learning Rate: {lr}'
                )
            plt.xlabel('Depth')
            plt.ylabel('Accuracy')
            plt.title('Depth vs Accuracy')
            plt.legend()
            plt.grid(True)
            plt.show()
            # model = Sequential([
            #     Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(100, 100, 3)),
            #     MaxPooling2D(pool_size=(2, 2)),
            #     Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
            #     MaxPooling2D(pool_size=(2, 2)),
            #     UpSampling2D(size=(2, 2)),
            #     Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
            #     UpSampling2D(size=(2, 2)),
            #     Conv2D(1, kernel_size=(1, 1), activation='sigmoid', padding='same')  # Change output layer to 1 channel and sigmoid
            # ])
            # model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
            
            model.save("cnn_pixelwise_model_2.keras")
            print("cnn_pixelwise_model.keras")
        else:
            X, y = [], []

            for img_path in self.image_list:
                img_array = self.preprocess_image(img_path)  # Preprocess image for training

                # Look for a saved mask corresponding to this image with .png extension
                mask_path = img_path.replace(".jpg", "_label.png")  # Assuming mask file naming pattern with .png extension
                if not os.path.exists(mask_path):
                    print(f"No saved mask found for {img_path}. Skipping.")
                    continue

                # Load the saved mask
                mask = Image.open(mask_path).convert("L")  # Convert to grayscale (L mode)

                mask_resized = mask.resize((100, 100), Image.NEAREST)
                mask_array = np.array(mask_resized)

                # Convert 255 to 1 (foreground) and 0 for background
                mask_array = np.where(mask_array == 255, 1, 0)

                # Preprocess image and mask data to create pixel-level input data
                pixel_data = self.pixelwise_data_prep(img_array)
                X.extend(pixel_data)  # Add pixel-level data
                y.extend(mask_array.flatten())  # Flatten mask to match pixel data format

            # Convert X and y to numpy arrays for training
            X = np.array(X)
            y = np.array(y)

            # Ensure there’s data to train with
            if len(X) == 0 or len(y) == 0:
                print("No training data available. Please ensure images and masks are present.")
                return

            # Split the dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            if model_choice == "Random Forest":
                rf = RandomForestClassifier()
                param_dist = {
                    'n_estimators': randint(50, 500),
                    'max_depth': randint(10, 50),
                    'min_samples_split': randint(2, 20),
                    'min_samples_leaf': randint(1, 20)
                }

                # Perform RandomizedSearchCV
                random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, 
                                                   n_iter=5, scoring='accuracy', cv=3, 
                                                   verbose=1, random_state=42, n_jobs=-1)

                # Fit the model
                random_search.fit(X_train, y_train)
                model = random_search.best_estimator_
                # Convert cv_results_ to a DataFrame for easy plotting
                results = pd.DataFrame(random_search.cv_results_)

                # Extract the parameters and mean test scores
                n_estimators = results['param_n_estimators']
                max_depth = results['param_max_depth']
                mean_test_score = results['mean_test_score']

                # Plotting
                plt.figure(figsize=(12, 8))
                scatter = plt.scatter(n_estimators, mean_test_score, c=max_depth, cmap='viridis', edgecolor="k", s=100)
                plt.colorbar(scatter, label="Max Depth")
                plt.xlabel("Number of Estimators (n_estimators)")
                plt.ylabel("Mean Test Accuracy")
                plt.title("Randomized Search Results for Random Forest Hyperparameters")
                plt.grid(True)
                plt.show()
            else:  # Logistic Regression
                lr = LogisticRegression(random_state=42, max_iter=1000)

                # Define the hyperparameter distribution for RandomizedSearchCV
                param_dist = {
                    'C': loguniform(0.001, 100),       # Inverse of regularization strength (sampled log-uniformly)
                    'penalty': ['l1', 'l2'],           # Regularization type (only 'l1' and 'l2' work with solvers like 'liblinear' and 'saga')
                    'solver': ['liblinear', 'saga']    # Solvers that support both 'l1' and 'l2' penalty
                }

                # Perform RandomizedSearchCV
                random_search = RandomizedSearchCV(estimator=lr, param_distributions=param_dist, 
                                                   n_iter=10, scoring='accuracy', cv=3, 
                                                   verbose=1, random_state=42, n_jobs=-1)

                # Fit the model
                random_search.fit(X_train, y_train)
                print("Best parameters found:")
                print(random_search.best_params_)
                model = random_search.best_estimator_
                # Convert cv_results_ to a DataFrame for easy plotting
                results = pd.DataFrame(random_search.cv_results_)

                # Extract the parameters and mean test scores
                C_values = results['param_C']
                penalty_types = results['param_penalty']
                mean_test_score = results['mean_test_score']

                l1_scores = mean_test_score[penalty_types == 'l1']
                l2_scores = mean_test_score[penalty_types == 'l2']
                l1_C_values = C_values[penalty_types == 'l1']
                l2_C_values = C_values[penalty_types == 'l2']

                # Plotting
                plt.figure(figsize=(12, 8))

                # Plot for 'l1' penalty
                plt.plot(l1_C_values, l1_scores, marker='o', color='blue', label='l1 Penalty')
                # Plot for 'l2' penalty
                plt.plot(l2_C_values, l2_scores, marker='o', color='green', label='l2 Penalty')

                # Log scale for C values
                plt.xscale('log')
                plt.xlabel("Inverse Regularization Strength (C)")
                plt.ylabel("Mean Test Accuracy")
                plt.title("Mean Test Accuracy vs Regularization Strength for Logistic Regression Penalties")
                plt.legend(title="Penalty")
                plt.grid(True)
                plt.show()

                print("Logistic Regression model training completed.")
            # Evaluate the model's performance
            accuracy = model.score(X_test, y_test)
            print(f"Model trained with accuracy: {accuracy}")

            # Save the model
            model_filename = "C:/Users/Asus/OneDrive/Desktop/ML/model_pixelwise.pkl"
            with open(model_filename, "wb") as f:
                joblib.dump(model, f)
            print(f"Model saved as {model_filename}")


    def test_image(self):
        test_image_path = filedialog.askopenfilename(title="Select Test Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if test_image_path:
            test_image_array = self.preprocess_image(test_image_path)
            model_choice = self.model_var.get()

            if model_choice == "CNN":
                import tensorflow as tf
                # CNN Model
                model = tf.keras.models.load_model("cnn_pixelwise_model_2.keras")
                predictions = model.predict(np.expand_dims(test_image_array, axis=0))
                predicted_label = (predictions > 0.5).astype(np.uint8).reshape(100, 100)
                
                # Convert prediction to an image and display
                label_image_pil = Image.fromarray((predicted_label * 255).astype(np.uint8), mode='L')
                self.show_prediction(label_image_pil.resize((500, 500)))
            
            else:
                # Load the pre-trained model using joblib (SVM, Random Forest, Logistic Regression)
                model_filename = "C:/Users/Asus/OneDrive/Desktop/ML/model_pixelwise.pkl"
                model = joblib.load(model_filename)

                # Prepare the image data for prediction (reshape to match the training data format)
                pixel_data = self.pixelwise_data_prep(test_image_array)  # Convert to the right shape for the classifier
                # pixel_data = pixel_data.reshape(1, -1)  # Reshape to match (1, num_pixels * 3)
                pixel_data = np.array(pixel_data)
                # Predict using the trained model
                prediction = model.predict(pixel_data)

                # Reshape prediction to match the image size
                predicted_label = prediction.reshape(100, 100)

                # Convert prediction to an image and display
                label_image_pil = Image.fromarray((predicted_label * 255).astype(np.uint8), mode='L')
                self.show_prediction(label_image_pil.resize((500, 500)))

    # (Other methods, such as `preprocess_image` and `show_prediction`, remain unchanged)
    def show_prediction(self, combined_image):
        # Create a new window to display the labeled image
        prediction_window = tk.Toplevel(self.root)
        prediction_window.title("Prediction Result")
        
        tk_image = ImageTk.PhotoImage(combined_image)
        canvas = tk.Canvas(prediction_window, width=500, height=500)
        canvas.pack()
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
        prediction_window.mainloop()
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageLabelingApp(root)
    root.mainloop()
