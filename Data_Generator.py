import tkinter as tk
from tkinter import filedialog, messagebox
import os
from PIL import Image
import torch
from torchvision import transforms
import shutil

# Define the augmentation pipeline
transform = transforms.Compose([
    transforms.RandomAffine(degrees=(-5, 5), translate=(0.05, 0.05), scale=(0.9, 1.1), shear=5),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1)], p=0.3),
])

class ImageAugmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Augmentation Tool")
        self.root.geometry("500x500")

        # Variables
        self.image_paths = []  # List to store multiple image paths
        self.save_folder = None
        self.target_num = tk.StringVar()
        self.exclude_originals = tk.BooleanVar(value=True)  # Checkbox variable, default checked
        self.folder_name = tk.StringVar(value="augmented_images")  # Default folder name

        # GUI Elements
        tk.Label(root, text="Image Augmentation Tool", font=("Arial", 16)).pack(pady=10)

        # Upload Images Button
        tk.Button(root, text="Upload Images", command=self.upload_images).pack(pady=5)

        # Image Path Label
        self.image_path_label = tk.Label(root, text="No images uploaded", wraplength=400)
        self.image_path_label.pack(pady=5)

        # Target Number Entry
        tk.Label(root, text="Enter target number of images:").pack(pady=5)
        tk.Entry(root, textvariable=self.target_num).pack(pady=5)

        # Checkbox for excluding original images
        tk.Checkbutton(root, text="Exclude original images (all images will be augmented)", 
                       variable=self.exclude_originals).pack(pady=5)

        # Folder name entry
        tk.Label(root, text="Output folder name:").pack(pady=5)
        tk.Entry(root, textvariable=self.folder_name).pack(pady=5)

        # Choose Save Folder Button
        tk.Button(root, text="Choose Save Folder", command=self.choose_save_folder).pack(pady=5)

        # Save Folder Label
        self.save_folder_label = tk.Label(root, text="No folder selected", wraplength=400)
        self.save_folder_label.pack(pady=5)

        # Generate Button
        tk.Button(root, text="Generate Images", command=self.generate_images).pack(pady=20)

    def upload_images(self):
        """Function to upload multiple images."""
        self.image_paths = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if self.image_paths:
            self.image_path_label.config(text=f"Uploaded {len(self.image_paths)} images")
        else:
            self.image_path_label.config(text="No images uploaded")

    def choose_save_folder(self):
        """Function to choose a save folder."""
        self.save_folder = filedialog.askdirectory(title="Select Folder to Save Images")
        if self.save_folder:
            self.save_folder_label.config(text=f"Save Folder: {self.save_folder}")
        else:
            self.save_folder_label.config(text="No folder selected")

    def generate_images(self):
        """Function to generate augmented images."""
        if not self.image_paths:
            messagebox.showerror("Error", "Please upload at least one image.")
            return
        if not self.save_folder:
            messagebox.showerror("Error", "Please select a save folder.")
            return
        
        # Check if folder name is provided
        folder_name = self.folder_name.get().strip()
        if not folder_name:
            messagebox.showerror("Error", "Please enter a folder name.")
            return
            
        try:
            target_num = int(self.target_num.get())
            if not self.exclude_originals.get() and target_num < len(self.image_paths):
                messagebox.showerror("Error", f"Target number of images must be at least {len(self.image_paths)} (number of uploaded images) when including originals.")
                return
            if self.exclude_originals.get() and target_num < 1:
                messagebox.showerror("Error", "Target number of images must be at least 1 when excluding originals.")
                return
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for target images.")
            return

        # Create a new folder inside the selected save folder using the user-provided name
        new_folder_path = os.path.join(self.save_folder, folder_name)
        
        # Check if folder already exists
        if os.path.exists(new_folder_path):
            response = messagebox.askyesno("Warning", f"Folder '{folder_name}' already exists. Do you want to overwrite it?")
            if response:
                # User agreed to overwrite
                shutil.rmtree(new_folder_path)
            else:
                # User declined to overwrite
                return
                
        os.makedirs(new_folder_path, exist_ok=True)

        original_images = [Image.open(image_path).convert("RGB") for image_path in self.image_paths]

        if not self.exclude_originals.get():
            # Include original images and generate the rest via augmentation
            # Copy the original images to the new folder as img1.jpg, img2.jpg, etc.
            for i, image_path in enumerate(self.image_paths):
                original_image_name = f"img{i + 1}.jpg"
                original_image_path = os.path.join(new_folder_path, original_image_name)
                shutil.copy(image_path, original_image_path)

            # Generate augmented images
            num_to_generate = target_num - len(self.image_paths)  # Number of augmented images needed
            current_image_index = 0  # Index to cycle through original images

            for i in range(num_to_generate):
                # Select the next image in the cycle
                image = original_images[current_image_index]

                # Apply augmentation
                augmented_image = transform(image)

                # Save the augmented image
                augmented_image_name = f"img{len(self.image_paths) + i + 1}.jpg"
                augmented_image_path = os.path.join(new_folder_path, augmented_image_name)
                augmented_image.save(augmented_image_path)

                # Move to the next image in the cycle
                current_image_index = (current_image_index + 1) % len(self.image_paths)

            messagebox.showinfo("Success", f"Generated {target_num} images (including {len(self.image_paths)} originals) in {new_folder_path}")
        else:
            # Exclude original images, generate all via augmentation
            num_to_generate = target_num  # All images are augmented
            current_image_index = 0  # Index to cycle through original images

            for i in range(num_to_generate):
                # Select the next image in the cycle
                image = original_images[current_image_index]

                # Apply augmentation
                augmented_image = transform(image)

                # Save the augmented image
                augmented_image_name = f"img{i + 1}.jpg"
                augmented_image_path = os.path.join(new_folder_path, augmented_image_name)
                augmented_image.save(augmented_image_path)

                # Move to the next image in the cycle
                current_image_index = (current_image_index + 1) % len(self.image_paths)

            messagebox.showinfo("Success", f"Generated {target_num} augmented images (no originals included) in {new_folder_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAugmentationApp(root)
    root.mainloop()
