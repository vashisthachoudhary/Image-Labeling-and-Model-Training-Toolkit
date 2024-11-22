import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk, ImageDraw
from datetime import datetime
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
import cv2
from tensorflow.keras.models import load_model
import albumentations as A
import tensorflow as tf
from xgboost import XGBClassifier
# from albumentations.pytorch.transforms import ToTensorV2  # For PyTorch compatibility
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
Resolution = 256
augmentation = A.Compose([
    # Geometric transformations
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=30, p=0.8),
    # A.ElasticTransform(alpha=1.0, sigma=50, p=0.5),
    # A.RandomCrop(height=Resolution/2, width=Resolution/2, p=0.5),  # Random cropping to 256x256
    # A.Perspective(scale=(0.05, 0.1), p=0.4),    # Perspective transform
    # Color and contrast adjustments
    # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
    # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
    # A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    # Blurring and noise
    # A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    # A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    # Coarse dropout (similar to CutOut)
    # A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=1, min_height=16, min_width=16, p=0.4),
    # Final transformation to tensor
    # ToTensorV2()
])
class ImageLabelingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Labeling and Model Training Tool")

        # Initialize canvas and image variables
        self.canvas = tk.Canvas(root, width=500, height=500)
        self.canvas.grid(row=0, column=0, columnspan=10, pady=10)  # Make canvas take full width

        self.image = None
        self.draw = None
        self.label_image = None  # For storing the labeled image
        self.original_image = None  # Store original image
        self.current_image_index = 0
        self.image_list = []
        self.labels = []  # To store labels (painted masks)
        self.history = []  # To store history of label images for undo

        # Buttons and other widgets in one row
        self.load_btn = tk.Button(root, text="Load Images", command=self.load_images)
        self.load_btn.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        self.next_btn = tk.Button(root, text="Next Image", command=self.next_image)
        self.next_btn.grid(row=1, column=1, padx=10, pady=10, sticky="w")

        self.train_btn = tk.Button(root, text="Train Model", command=self.train_model)
        self.train_btn.grid(row=1, column=2, padx=10, pady=10, sticky="w")

        self.model_button = tk.Button(root, text="Select Model", command=self.select_model)
        self.model_button.grid(row=1, column=3, padx=10, pady=10, sticky="w")

        self.test_btn = tk.Button(root, text="Test Image", command=self.test_image)
        self.test_btn.grid(row=1, column=4, padx=10, pady=10, sticky="w")

        self.test_btn_live = tk.Button(root, text="Live Video", command=self.live_segmentation)
        self.test_btn_live.grid(row=1, column=5, padx=10, pady=10, sticky="w")

        # Model selection dropdown
        self.model_var = tk.StringVar()
        self.model_var.set("CNN")  # Default value
        self.model_dropdown = ttk.Combobox(root, textvariable=self.model_var, values=["CNN", "Random Forest", "Logistic Regression", "XGBoost"])
        self.model_dropdown.grid(row=1, column=6, padx=10, pady=10)

        # Brush size selection
        self.brush_size_var = tk.IntVar(value=10)  # Default brush size
        self.brush_size_slider = tk.Scale(root, from_=1, to=50, orient=tk.HORIZONTAL, label="Brush Size", variable=self.brush_size_var)
        self.brush_size_slider.grid(row=1, column=7, padx=10, pady=10)

        # Hyperparameter tuning options
        self.hyperparam_var = tk.StringVar(value="No")  # Default value: No hyperparameter tuning
        self.hyperparam_label = tk.Label(root, text="Hyperparameter Tuning:")
        self.hyperparam_label.grid(row=1, column=8, padx=10, pady=10)

        self.hyperparam_no = tk.Radiobutton(root, text="No", variable=self.hyperparam_var, value="No")
        self.hyperparam_no.grid(row=1, column=9, padx=10, pady=10)

        self.hyperparam_yes = tk.Radiobutton(root, text="Yes", variable=self.hyperparam_var, value="Yes")
        self.hyperparam_yes.grid(row=1, column=10, padx=10, pady=10)

        # Set up drawing functionality (left click to draw, right click to erase)
        self.canvas.bind("<B1-Motion>", self.paint)  # While dragging with the mouse pressed
        self.canvas.bind("<Button-1>", self.paint)    # On mouse click
        self.canvas.bind("<Button-3>", self.eraser)   # Right-click for eraser
        self.canvas.bind("<B3-Motion>", self.eraser)  # Right-click for eraser while dragging

  

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

    # def undo(self):
    #     if self.history:
    #         self.history.pop()  # Remove the last image from history
    #         if self.history:
    #             self.label_image = self.history[-1].copy()  # Get the last image in history
    #         else:
    #             self.label_image = Image.new("RGBA", self.image.size, (255, 255, 255, 0))  # Reset to a new transparent image
    #         # Update the displayed image
    #         combined_image = Image.alpha_composite(self.image, self.label_image)
    #         self.update_canvas(combined_image)

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
            label_array_resized = np.array(label_output.resize((Resolution, Resolution)))  # Resize for consistency
            self.labels.append(label_array_resized)  # Append the label for the image
            save_path = os.path.splitext(self.image_list[self.current_image_index])[0] + "_label.png"  # Save as PNG
            label_output.save(save_path)  # Save the label image as .png
            print(f"Auto-saved label image: {save_path}")

    def select_model(self):
        """
        Open a file dialog to select a model file. The path is stored for later use.
        """
        model_file_path = filedialog.askopenfilename(
            title="Select Model File", 
            filetypes=[("Keras Model Files", "*.keras"), ("Pickle Files", "*.pkl")]
        )
        if model_file_path:
            self.model_path = model_file_path  # Store the selected model file path
            print(f"Selected model file: {self.model_path}")


    def preprocess_image(self, image_path, target_size=(Resolution, Resolution)):
        
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
    

    def augment_data(self, images, segmentation_maps, aug_count=2):
        augmented_images = []
        augmented_masks = []

        # Apply augmentation to each image and mask in the batch
        for img, mask in zip(images, segmentation_maps):
            for _ in range(aug_count):
                # Apply augmentation to the image and mask
                augmented = augmentation(image=img, mask=mask)
                augmented_images.append(augmented['image'])
                augmented_masks.append(augmented['mask'])

        # Convert lists to numpy arrays
        augmented_images = np.array(augmented_images)
        augmented_masks = np.array(augmented_masks)

        # print(f"Augmented data shape: {augmented_images.shape[0]}")  # This should always be batch_size * aug_count

        return augmented_images, augmented_masks



    def data_generator(self, image_paths, mask_paths, batch_size=8, target_size=(512, 512), augment=True, aug_count=2):
        """
        Generator that yields batches of images and masks with augmentation.
        Augmented images are added to the training data, not replacing the originals.

        Args:
            image_paths (list): List of paths to image files.
            mask_paths (list): List of paths to mask files.
            batch_size (int): Number of original images per batch.
            target_size (tuple): Desired size of output images and masks.
            augment (bool): Whether to apply data augmentation.
            aug_count (int): Number of augmented samples to add per image.

        Yields:
            tuple: (batch_images, batch_masks), where both are numpy arrays.
        """

        while True:
            length = len(image_paths)
            for i in range(0, length, batch_size):
                original_images = []
                original_masks = []
                augmented_images = []
                augmented_masks = []

                # Load and preprocess the original batch
                for img_path, mask_path in zip(image_paths[i:i+batch_size], mask_paths[i:i+batch_size]):
                    # Load and preprocess the image
                    img = Image.open(img_path).convert("RGB").resize(target_size)
                    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]

                    # Load and preprocess the mask
                    mask = Image.open(mask_path).convert("L").resize(target_size, Image.NEAREST)
                    mask_array = np.where(np.array(mask) == 255, 0, 1).astype(np.float32)
                    mask_array = np.expand_dims(mask_array, axis=-1)  # Ensure (H, W, 1)
                    # Convert back to a PIL Image for saving
                    # mask.save(r"C:\Users\Asus\OneDrive\Desktop\testing\ImageLabelingAndTrainingToolkit\temp\processed_mask2.png")
                    # label_image_pil = Image.fromarray((mask_array.squeeze() * 255).astype(np.uint8), mode='L')
                    
                    # Save the processed mask with a complete path
                    # save_path = r"C:\Users\Asus\OneDrive\Desktop\testing\ImageLabelingAndTrainingToolkit\temp\processed_mask.png"
                    # label_image_pil.save(save_path)
                    
                    # Append original images and masks
                    original_images.append(img_array)
                    original_masks.append(mask_array)

                    # Generate augmented versions
                    if augment:
                        aug_images, aug_masks = self.augment_data([img_array], [mask_array], aug_count=aug_count)
                        augmented_images.extend(aug_images)
                        augmented_masks.extend(aug_masks)

                # Combine originals and augmented data
                all_images = np.concatenate([original_images, augmented_images], axis=0)
                all_masks = np.concatenate([original_masks, augmented_masks], axis=0)

                # print(f"Total batch size: {len(all_images)}")

                # Yield in fixed-size batches
                for j in range(0, len(all_images), batch_size):
                    yield all_images[j:j+batch_size], all_masks[j:j+batch_size]


    def save_model_with_user_input(self, model, default_filename="model.keras"):
        """
        Save the trained model to a location specified by the user.

        Args:
            model: Trained model to save.
            default_filename: Default name for the model file if the user doesn't modify it.
        """
        # Open a file dialog to let the user specify the file location and name
        filepath = filedialog.asksaveasfilename(
            defaultextension=".keras",  # Default file extension
            filetypes=[("Keras Model", "*.keras"), ("All Files", "*.*")],  # Allowed file types
            initialfile=default_filename,  # Default filename
            title="Save Model As"
        )

        if filepath:  # If the user selects a path
            # Save the model
            model.save(filepath)  # For Keras models; adjust for other model types
            print(f"Model saved as: {filepath}")
        else:
            print("Model save canceled by user.")

    def train_model(self):
        model_choice = self.model_var.get()
        print(f"{bcolors.OKBLUE}Selected model {model_choice}{bcolors.ENDC}")
        use_hyperparam_tuning = self.hyperparam_var.get() == "Yes"
        print(f"{bcolors.OKBLUE}Hyperparameter tuning: {use_hyperparam_tuning}{bcolors.ENDC}")
        if model_choice == "CNN":
            mask_paths,image_paths = [], []

            for img_path in self.image_list:
                mask_path = img_path.replace(".jpeg", "_label.png")  # Assuming mask file naming pattern with .png extension
                mask_paths.append(mask_path)
                image_paths.append(img_path)
            print(f"{bcolors.OKBLUE}Extracted Images & Masks{bcolors.ENDC}")

            # Train CNN
            def create_cnn_model(learning_rate = 0.0001, filters=32, kernel_size=(3, 3), depth=6):
                model = Sequential()

                # Initial Conv and Pool layers
                model.add(Conv2D(filters, kernel_size=kernel_size, activation='relu', padding='same', input_shape=(Resolution, Resolution, 3)))
                model.add(MaxPooling2D(pool_size=(2, 2)))

                # Adding Conv-Pool layers based on depth
                for i in range(1, depth):
                    model.add(Conv2D(filters * (2 ** i), kernel_size=kernel_size, activation='relu', padding='same'))

                    # Only add MaxPooling if dimensions are larger than 1x1
                    model.add(MaxPooling2D(pool_size=(2, 2)))

                    # Check if spatial dimension will reach 1x1, stop if true
                    if (Resolution // (2 ** (i+1))) < 2:
                        break
                    
                # UpSampling and Conv layers to restore image dimensions
                for i in range(depth, 1, -1):
                    model.add(Conv2D(filters * (2 ** i), kernel_size=kernel_size, activation='relu', padding='same'))
                    model.add(UpSampling2D(size=(2, 2)))
                    if (Resolution // (2 ** (i+1))) < 2:
                        break                    

                model.add(UpSampling2D(size=(2, 2)))
                model.add(Conv2D(1, kernel_size=(1, 1), activation='sigmoid',padding='same'))  # Output layer with 1 channel

                # Compile the model
                model.compile(optimizer=Adam(learning_rate = learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
                return model
            
            def hyperparameter_tuning(train_gen, val_gen,steps_per_epoch,validation_steps):
                print(f"{bcolors.OKBLUE}Hyperparameter tuning{bcolors.ENDC}")
                acc_results = []
                best_model = None
                max_accuracy = 0

                for lr in [0.01, 0.001, 0.0001]:
                    for filters in [16, 32, 64]:
                        for kernel_size in [(3, 3), (5, 5), (7, 7)]:
                            for depth in range(1, 4):  # Limit depth to avoid excessive computation
                                try:
                                    # Create the CNN model
                                    model = create_cnn_model(learning_rate=lr, filters=filters, kernel_size=kernel_size, depth=depth)
                                    
                                    # Train the model with train_gen and validate on val_gen
                                    history = model.fit(
                                        train_gen,
                                        steps_per_epoch=steps_per_epoch,
                                        validation_data=val_gen,
                                        validation_steps=validation_steps,
                                        epochs=10,
                                        batch_size=8,
                                    )


                                    # Get the best validation accuracy
                                    val_accuracy = max(history.history['val_accuracy'])
                                    print(f"{bcolors.OKCYAN}model with learning rate: {lr} filter: {filters} kernel_size: {kernel_size} depth: {depth} has accuracy: {val_accuracy} {bcolors.ENDC}")
                                    acc_results.append((lr, filters, kernel_size, depth, val_accuracy))
            
                                    # Update the best model
                                    if val_accuracy > max_accuracy:
                                        max_accuracy = val_accuracy
                                        best_model = model
                                except Exception as e:
                                    print(f"Error with combination: lr={lr}, filters={filters}, kernel={kernel_size}, depth={depth}")
                                    print(e)
                                    continue
                                
                print(f"Best model found with Validation Accuracy: {max_accuracy}")
                print(f"Learning Rate: {lr}, Filters: {filters}, Kernel Size: {kernel_size}, Depth: {depth}")
                return best_model, acc_results


            def plot_hyperparameter_combinations(acc):
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

            batch_size = 8
            model_CNN = None
            if use_hyperparam_tuning:
                print(bcolors.OKGREEN + "Hyperparameter Tuning for CNN" + bcolors.ENDC)
                # print("Using hyperparameter tuning for CNN.")
                train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(
                    image_paths, mask_paths, test_size=0.2, random_state=42
                )
                print(bcolors.OKGREEN + "train test splited" + bcolors.ENDC)
                train_gen = self.data_generator(
                    image_paths=train_img_paths, mask_paths=train_mask_paths, batch_size=batch_size, target_size=(Resolution, Resolution)
                )
                print(bcolors.OKGREEN + "train gen created" + bcolors.ENDC)
                val_gen = self.data_generator(
                    image_paths=val_img_paths, mask_paths=val_mask_paths, batch_size=batch_size, target_size=(Resolution, Resolution)
                )
                print(bcolors.OKGREEN + "val gen created" + bcolors.ENDC)

                No_of_aug = 2

                steps_per_epoch = (len(train_img_paths) * (No_of_aug + 1)) // batch_size  # e.g., 200 // 8 = 25
                validation_steps = len(val_img_paths) // batch_size  # similarly for validation data

                model, acc = hyperparameter_tuning(train_gen,val_gen,steps_per_epoch,validation_steps)

                model.fit(
                    train_gen,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=val_gen,
                    validation_steps=validation_steps,
                    epochs=10,
                    batch_size=8,
                )
                print("Model training completed.")
                plot_hyperparameter_combinations(acc)
                print(f"Accuracy: {acc[4]}")
                
                model_CNN = model
            else:
                print("Training CNN without hyperparameter tuning.")
                model = create_cnn_model()

                train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(
                    image_paths, mask_paths, test_size=0.2, random_state=42
                )
                print(bcolors.OKGREEN + "train test splited" + bcolors.ENDC)
                train_gen = self.data_generator(
                    image_paths=train_img_paths, mask_paths=train_mask_paths, batch_size=batch_size, target_size=(Resolution, Resolution)
                )
                print(bcolors.OKGREEN + "train gen created" + bcolors.ENDC)
                val_gen = self.data_generator(
                    image_paths=val_img_paths, mask_paths=val_mask_paths, batch_size=batch_size, target_size=(Resolution, Resolution)
                )
                print(bcolors.OKGREEN + "val gen created" + bcolors.ENDC)

                No_of_aug = 2

                steps_per_epoch = (len(train_img_paths) * (No_of_aug + 1)) // batch_size  # e.g., 200 // 8 = 25
                validation_steps = len(val_img_paths) // batch_size  # similarly for validation data

                model.fit(
                    train_gen,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=val_gen,
                    validation_steps=validation_steps,
                    epochs=10,
                    batch_size=64,
                )
                print("Model training completed.")
                model_CNN = model

            if model_CNN:
                default_filename = f"cnn_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
                self.save_model_with_user_input(model_CNN, default_filename)


        else:
            model_choice = self.model_var.get()
            print(f"{bcolors.OKBLUE}Selected model {model_choice}{bcolors.ENDC}")
            use_hyperparam_tuning = self.hyperparam_var.get() == "Yes"
            print(f"{bcolors.OKBLUE}Hyperparameter tuning: {use_hyperparam_tuning}{bcolors.ENDC}")
            def hyperparameter_tuning_xgb(X_train, y_train):
                param_dist = {
                    'n_estimators': randint(50, 300),
                    'max_depth': randint(3, 10),
                    'learning_rate': loguniform(0.01, 0.3),
                    'subsample': loguniform(0.5, 1.0),
                    'colsample_bytree': loguniform(0.5, 1.0),
                }
                xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
                random_search = RandomizedSearchCV(
                    estimator=xgb, param_distributions=param_dist, n_iter=10,
                    scoring='accuracy', cv=3, verbose=1, random_state=42, n_jobs=-1
                )
                random_search.fit(X_train, y_train)
                return random_search.best_estimator_, random_search.cv_results_
            

            def data_augmentation(img_array, mask_array, num_augmentations=2):
                augmented_images, augmented_masks = [], []

                for _ in range(num_augmentations):
                    if img_array.shape[:2] != mask_array.shape:
                        mask_resized = Image.fromarray(mask_array).resize(
                            (img_array.shape[1], img_array.shape[0]), Image.NEAREST
                        )
                        mask_array = np.array(mask_resized)

                    augmented = augmentation(image=img_array, mask=mask_array)
                    augmented_images.append(augmented['image'])
                    augmented_masks.append(augmented['mask'])

                return augmented_images, augmented_masks


            def preprocess_data(image_paths, mask_paths):
                X, y = [], []
                for img_path, mask_path in zip(image_paths, mask_paths):
                    img_array = self.preprocess_image(img_path)
                    mask = Image.open(mask_path).convert("L")
                    mask_resized = mask.resize((Resolution, Resolution), Image.NEAREST)
                    mask_array = np.array(mask_resized)
                    mask_array = np.where(mask_array == 255, 0, 1)

                    pixel_data = self.pixelwise_data_prep(img_array)
                    augmented_images, augmented_masks = data_augmentation(img_array, mask_array)
                    X.extend(pixel_data)
                    y.extend(mask_array.flatten())

                    for aug_image, aug_mask in zip(augmented_images, augmented_masks):
                        aug_pixel_data = self.pixelwise_data_prep(aug_image)
                        X.extend(aug_pixel_data)
                        y.extend(aug_mask.flatten())
                return np.array(X), np.array(y)

            def hyperparameter_tuning_rf(X_train, y_train):
                param_dist = {
                    'n_estimators': randint(50, 500),
                    'max_depth': randint(10, 50),
                    'min_samples_split': randint(2, 20),
                    'min_samples_leaf': randint(1, 20)
                }
                rf = RandomForestClassifier()
                random_search = RandomizedSearchCV(
                    estimator=rf, param_distributions=param_dist, n_iter=10,
                    scoring='accuracy', cv=3, verbose=1, random_state=42, n_jobs=-1
                )
                random_search.fit(X_train, y_train)
                return random_search.best_estimator_, random_search.cv_results_

            def hyperparameter_tuning_lr(X_train, y_train):
                param_dist = {
                    'C': loguniform(0.001, 100),
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
                lr = LogisticRegression(max_iter=10000, random_state=42)
                random_search = RandomizedSearchCV(
                    estimator=lr, param_distributions=param_dist, n_iter=10,
                    scoring='accuracy', cv=3, verbose=1, random_state=42, n_jobs=-1
                )
                random_search.fit(X_train, y_train)
                return random_search.best_estimator_, random_search.cv_results_

            if model_choice in ["Random Forest", "Logistic Regression", "XGBoost"]:
                mask_paths, image_paths = [], []
                for img_path in self.image_list:
                    mask_path = img_path.replace(".jpeg", "_label.png")
                    if os.path.exists(mask_path):
                        mask_paths.append(mask_path)
                        image_paths.append(img_path)
                    else:
                        print(f"No mask found for {img_path}. Skipping.")
                if not image_paths:
                    print("No valid images found. Exiting.")
                    return

                X, y = preprocess_data(image_paths, mask_paths)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                if model_choice == "Random Forest":
                    if use_hyperparam_tuning:
                        model, cv_results = hyperparameter_tuning_rf(X_train, y_train)
                    else:
                        model = RandomForestClassifier().fit(X_train, y_train)
                elif model_choice == "Logistic Regression":
                    if use_hyperparam_tuning:
                        model, cv_results = hyperparameter_tuning_lr(X_train, y_train)
                    else:
                        model = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)
                elif model_choice == "XGBoost":
                    if use_hyperparam_tuning:
                        model, cv_results = hyperparameter_tuning_xgb(X_train, y_train)
                    else:
                        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42).fit(X_train, y_train)

                accuracy = model.score(X_test, y_test)
                print(f"Model trained with accuracy: {accuracy}")

                # Save model
                model_filename = f"{model_choice.lower()}_model.pkl"
                with open(model_filename, "wb") as f:
                    joblib.dump(model, f)
                print(f"Model saved as {model_filename}")

                if use_hyperparam_tuning:
                    self.plot_hyperparameter_results(cv_results)


    def live_segmentation(self):
        # Load the trained CNN model
        model = load_model(r"cnn_pixelwise_model_Human.keras")

        # Open a connection to the camera (0 is usually the default camera)
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        print("Starting live segmentation. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Resize and preprocess the frame
            original_size = frame.shape[:2]  # Store original frame size (height, width)
            resized_frame = cv2.resize(frame, (Resolution, Resolution))  # Resize to the input size expected by the model
            input_frame = resized_frame / 255.0  # Normalize
            input_frame = np.expand_dims(input_frame, axis=0)  # Add batch dimension

            # Predict the segmentation mask
            predictions = model.predict(input_frame)
            predicted_mask = (predictions[0, :, :, 0] > 0.5).astype(np.uint8)  # Binary mask (threshold at 0.5)

            # Invert the mask if necessary
            inverted_mask = 1 - predicted_mask  # Flip the values (1 -> 0, 0 -> 1) to highlight the object

            # Resize the mask back to the original frame size
            mask_resized = cv2.resize(inverted_mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)

            # Overlay the mask on the original frame
            colored_mask = np.zeros_like(frame)
            colored_mask[:, :, 1] = mask_resized * 255  # Apply mask to the green channel (for visualization)
            overlaid_frame = cv2.addWeighted(frame, 0.7, colored_mask, 0.3, 0)

            # Display the overlaid frame using OpenCV
            cv2.imshow("Live Segmentation", overlaid_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()
    
    def test_image(self):
        if not self.model_path:
            print("Please select a model file first.")
            return

        # Allow the user to select the test image
        test_image_path = filedialog.askopenfilename(title="Select Test Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if test_image_path:
            test_image_array = self.preprocess_image(test_image_path)

            # Load the original image for visualization
            original_image = Image.open(test_image_path).resize((500, 500)).convert("RGBA")

            # Determine the model file type
            model_file = self.model_path
            file_extension = model_file.split('.')[-1]

            if file_extension == 'keras':
                # Load Keras model (e.g., CNN)
                model = tf.keras.models.load_model(model_file)
                predictions = model.predict(np.expand_dims(test_image_array, axis=0))
                predicted_label = (predictions > 0.5).astype(np.uint8).squeeze()

            elif file_extension == 'pkl':
                # Load other models (e.g., SVM, Random Forest, Logistic Regression)
                model = joblib.load(model_file)

                # Prepare the image data for prediction (reshape to match the training data format)
                pixel_data = self.pixelwise_data_prep(test_image_array)  # Convert to the right shape for the classifier
                pixel_data = np.array(pixel_data)

                # Predict using the trained model
                prediction = model.predict(pixel_data)

                # Reshape prediction to match the image size
                predicted_label = prediction.reshape(test_image_array.shape[:2])

            else:
                print("Unsupported model file type!")
                return
    
            # Create a green mask where the prediction is 1 (predicted object area)
            mask = Image.fromarray((predicted_label * 255).astype(np.uint8), mode='L').resize((500, 500))
    
            # Create a fully green image (RGBA) where green is applied only where the mask is 1
            green_mask = Image.new("RGBA", mask.size, (0, 255, 0, 0))  # Fully transparent image
            green_mask.paste((0, 255, 0, 128), mask=mask)  # Paste green with transparency only where mask is 1
    
            # Blend the green mask with the original image (only where the object is)
            blended_image = Image.alpha_composite(original_image.convert("RGBA"), green_mask)
    
            # Display the blended image
            self.show_prediction(blended_image)

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
