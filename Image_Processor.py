import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from pdf2image import convert_from_path
import cv2
import numpy as np
import shutil
import time
import os
import uuid

def load_pdf():
    """Load and display multiple one-page PDF files."""
    global pdf_files, pdf_image
    file_paths = filedialog.askopenfilenames(
        filetypes=[("PDF Files", "*.pdf")],
        title="Select one or more one-page PDF files"
    )
    if not file_paths:
        return
    pdf_files = list(file_paths)
    debug_images.clear()
    pdf_combobox['values'] = [os.path.basename(path) for path in pdf_files]
    if pdf_files:
        pdf_combobox.current(0)
        display_selected_pdf()

def display_selected_pdf():
    """Display the selected PDF from the combobox."""
    global pdf_image, pdf_files
    if not pdf_files:
        return
    
    selection = pdf_combobox.current()
    if selection < 0:
        return
        
    try:
        images = convert_from_path(pdf_files[selection], dpi=200)
        if len(images) != 1:
            messagebox.showerror("Error", "Please upload one-page PDF files.")
            return
        pdf_image = images[0]
        display_image(pdf_image)
        status_label.config(text=f"Loaded: {os.path.basename(pdf_files[selection])}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load PDF: {e}")

def display_image(image):
    """Display the given PIL image in the GUI."""
    global canvas, tk_image, scaled_image, image_scale, canvas_offset_x, canvas_offset_y, original_image
    canvas.delete("all")
    
    original_image = image
    image_width, image_height = image.size
    canvas_width, canvas_height = canvas.winfo_width(), canvas.winfo_height()
    if canvas_width <= 1:  # Canvas not yet drawn
        canvas_width, canvas_height = 600, 800
    
    # Calculate display scale to fit canvas
    image_scale = min(canvas_width / image_width, canvas_height / image_height)
    
    new_width = int(image_width * image_scale)
    new_height = int(image_height * image_scale)
    scaled_image = image.resize((new_width, new_height))
    
    # Center the image
    canvas_offset_x = (canvas_width - new_width) // 2
    canvas_offset_y = (canvas_height - new_height) // 2
    
    tk_image = ImageTk.PhotoImage(scaled_image)
    canvas.create_image(canvas_offset_x, canvas_offset_y, anchor="nw", image=tk_image)
    canvas.configure(scrollregion=(0, 0, canvas_width, canvas_height))

def start_crop(event):
    """Start selecting rectangle area for cropping."""
    global start_x, start_y, rect_id
    if rect_id is not None:
        canvas.delete(rect_id)
        rect_id = None
    
    canvas_x = canvas.canvasx(event.x)
    canvas_y = canvas.canvasy(event.y)
    start_x = canvas_x - canvas_offset_x
    start_y = canvas_y - canvas_offset_y
    rect_id = canvas.create_rectangle(canvas_x, canvas_y, canvas_x, canvas_y, outline="red", width=2)

def update_crop(event):
    """Update the cropping rectangle while dragging."""
    global rect_id
    canvas_x = canvas.canvasx(event.x)
    canvas_y = canvas.canvasy(event.y)
    canvas.coords(rect_id, start_x + canvas_offset_x, start_y + canvas_offset_y, canvas_x, canvas_y)

def finish_crop(event):
    """Finish cropping and store coordinates."""
    global crop_coords, rect_id
    if pdf_image is None:
        messagebox.showerror("Error", "No PDF image loaded.")
        return

    try:
        canvas_x = canvas.canvasx(event.x)
        canvas_y = canvas.canvasy(event.y)
        end_x = canvas_x - canvas_offset_x
        end_y = canvas_y - canvas_offset_y
        
        x1, y1, x2, y2 = min(start_x, end_x), min(start_y, end_y), max(start_x, end_x), max(start_y, end_y)
        x1 = max(0, min(x1, scaled_image.width))
        y1 = max(0, min(y1, scaled_image.height))
        x2 = max(0, min(x2, scaled_image.width))
        y2 = max(0, min(y2, scaled_image.height))
        
        x1_orig = int(x1 / image_scale)
        y1_orig = int(y1 / image_scale)
        x2_orig = int(x2 / image_scale)
        y2_orig = int(y2 / image_scale)
        
        crop_coords = (x1_orig, y1_orig, x2_orig, y2_orig)
        status_label.config(text=f"Selection complete. Region: {crop_coords}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to calculate crop coordinates: {e}")

def process_image(image_path):
    """Process the image to detect characters and prepare MNIST-formatted images, saving intermediate steps."""
    img = cv2.imread(image_path)
    
    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    
    # Step 2: Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 12
    )

    # Step 3: Remove vertical and horizontal lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    eroded = cv2.erode(binary, vertical_kernel, iterations=1)
    detected_vertical_lines = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, vertical_kernel)
    dilated_vertical_lines = cv2.dilate(detected_vertical_lines, vertical_kernel, iterations=1)
    
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    eroded_h = cv2.erode(binary, horizontal_kernel, iterations=1)
    detected_horizontal_lines = cv2.morphologyEx(eroded_h, cv2.MORPH_OPEN, horizontal_kernel)
    dilated_horizontal_lines = cv2.dilate(detected_horizontal_lines, horizontal_kernel, iterations=1)
    
    combined_lines = cv2.add(dilated_vertical_lines, dilated_horizontal_lines)
    cleaned_binary = cv2.subtract(binary, combined_lines)
    
    # Save a copy of cleaned_binary after line removal
    cleaned_binary_after_lines = cleaned_binary.copy()
    
    # Step 4: Clean up noise
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned_binary = cv2.morphologyEx(cleaned_binary, cv2.MORPH_CLOSE, closing_kernel)
    cleaned_binary = cv2.medianBlur(cleaned_binary, 3)

    def preprocess_to_mnist_format(image):
        """Convert an image to MNIST format (28x28, centered, grayscale)."""
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(image)
        if coords is None:
            return np.zeros((28, 28), dtype=np.uint8)
        x, y, w, h = cv2.boundingRect(coords)
        if w == 0 or h == 0:
            return np.zeros((28, 28), dtype=np.uint8)
        digit = image[y:y+h, x:x+w]
        max_dim = max(w, h)
        scale = 20.0 / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((28, 28), dtype=np.uint8)
        x_offset = (28 - new_w) // 2
        y_offset = (28 - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        return canvas

    margin = 5
    contours, _ = cv2.findContours(cleaned_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_dir = "/Users/ryan/Desktop/cropped_digits"
    os.makedirs(output_dir, exist_ok=True)

    img_height, img_width = cleaned_binary.shape

    def merge_nearby_contours(contours, distance_threshold=15):
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        merged_boxes = []
        used = [False] * len(bounding_boxes)

        for i in range(len(bounding_boxes)):
            if used[i]:
                continue
            x1, y1, w1, h1 = bounding_boxes[i]
            merged_xmin = x1
            merged_ymin = y1
            merged_xmax = x1 + w1
            merged_ymax = y1 + h1
            used[i] = True

            for j in range(i + 1, len(bounding_boxes)):
                if used[j]:
                    continue
                x2, y2, w2, h2 = bounding_boxes[j]
                dx = min(abs(x1 + w1 - x2), abs(x2 + w2 - x1)) if x1 < x2 + w2 and x2 < x1 + w1 else abs(x1 - x2)
                dy = min(abs(y1 + h1 - y2), abs(y2 + h2 - y1)) if y1 < y2 + h2 and y2 < y1 + h1 else abs(y1 - y2)
                distance = max(dx, dy)
                
                if distance < distance_threshold:
                    merged_xmin = min(merged_xmin, x2)
                    merged_ymin = min(merged_ymin, y2)
                    merged_xmax = max(merged_xmax, x2 + w2)
                    merged_ymax = max(merged_ymax, y2 + h2)
                    used[j] = True

            merged_boxes.append((merged_xmin, merged_ymin, merged_xmax - merged_xmin, merged_ymax - merged_ymin))
        return merged_boxes

    initial_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 2 < w < 100 and 15 < h < 100:
            if (x > margin and y > margin and 
                x + w < img_width - margin and 
                y + h < img_height - margin):
                initial_boxes.append(contour)

    bounding_boxes = merge_nearby_contours(initial_boxes, distance_threshold=10)
    bounding_boxes.sort(key=lambda box: box[0])

    mnist_images = []
    green_boxes_img = cv2.cvtColor(cleaned_binary.copy(), cv2.COLOR_GRAY2BGR)

    for i, (x, y, w, h) in enumerate(bounding_boxes):
        digit_roi = cleaned_binary[y:y+h, x:x+w]
        mnist_digit = preprocess_to_mnist_format(digit_roi)
        mnist_images.append(mnist_digit)
        cv2.rectangle(green_boxes_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert intermediate images to PIL format for display
    gray_pil = Image.fromarray(gray)
    enhanced_gray_pil = Image.fromarray(enhanced_gray)  # Add enhanced_gray
    binary_pil = Image.fromarray(binary)
    cleaned_binary_pil = Image.fromarray(cleaned_binary_after_lines)
    green_boxes_img_pil = Image.fromarray(green_boxes_img)

    return (
        cleaned_binary,  # Final processed image
        mnist_images,
        green_boxes_img,
        {
            'grayscale': gray_pil,
            'enhanced_grayscale': enhanced_gray_pil,  # Include enhanced_gray
            'binary': binary_pil,
            'line_removed': cleaned_binary_pil,
        }
    )

def process():
    """Process the cropped region and show debug window."""
    global crop_coords, pdf_files, rect_id, debug_images
    if pdf_image is None:
        messagebox.showerror("Error", "No PDF image loaded.")
        return
    if crop_coords is None:
        messagebox.showerror("Error", "Please draw a bounding box first.")
        return
    
    status_label.config(text="Processing... Please wait.")
    root.update()

    if rect_id is not None:
        canvas.delete(rect_id)
        rect_id = None

    debug_images.clear()
    output_dir = "/Users/ryan/Desktop/cropped_digits"
    
    for file_path in pdf_files:
        try:
            images = convert_from_path(file_path, dpi=200)
            if len(images) != 1:
                messagebox.showerror("Error", f"{os.path.basename(file_path)}: Not a one-page PDF.")
                continue
            original_img = images[0]
            cropped_image = original_img.crop(crop_coords)
            temp_image_path = "temp_cropped_region.png"
            cropped_image.save(temp_image_path)
            
            debug_entry = {
                'original': original_img,
                'cropped': cropped_image,
                'processed': None,
                'mnist_images': None,
                'green_boxes_img': None,
                'intermediate': {}
            }
            
            processed_img, mnist_images, green_boxes_img, intermediate_imgs = process_image(temp_image_path)
            processed_img_pil = Image.fromarray(processed_img)
            green_boxes_img_pil = Image.fromarray(green_boxes_img)

            debug_entry.update({
                'processed': processed_img_pil,
                'mnist_images': mnist_images,
                'green_boxes_img': green_boxes_img_pil,
                'intermediate': intermediate_imgs
            })
            
            debug_images[file_path] = debug_entry
            
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
                
        except Exception as e:
            messagebox.showerror("Error", f"{os.path.basename(file_path)}: Error processing file - {e}")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    status_label.config(text="Processing complete!")
    if debug_images:
        show_debug_window(list(debug_images.keys())[0])

def download_image(image, filename_prefix, parent_window, file_path=None):
    """Prompt user to save a single image."""
    if not image:
        messagebox.showinfo("Download Error", f"No {filename_prefix} image available to download.")
        return
    
    save_dir = filedialog.askdirectory(title=f"Select Folder to Save {filename_prefix} Image", parent=parent_window)
    if not save_dir:
        return
    
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        pdf_basename = os.path.splitext(os.path.basename(file_path))[0] if file_path else "unknown"
        filename = os.path.join(save_dir, f"{filename_prefix}_{pdf_basename}_{timestamp}.png")
        image.save(filename)
        messagebox.showinfo("Download Complete", f"{filename_prefix} image saved successfully to:\n{filename}")
    except Exception as e:
        messagebox.showerror("Download Error", f"Failed to save {filename_prefix} image: {e}")

def download_mnist_images(debug_data, parent_window, file_path=None):
    """Prompt user to save MNIST-formatted images."""
    if 'mnist_images' not in debug_data or not debug_data['mnist_images']:
        messagebox.showinfo("Download Error", "No MNIST-formatted images available to download.")
        return
    
    save_dir = filedialog.askdirectory(title="Select Folder to Save MNIST Images", parent=parent_window)
    if not save_dir:
        return
    
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        pdf_basename = os.path.splitext(os.path.basename(file_path))[0] if file_path else "unknown"
        for i, mnist_img in enumerate(debug_data['mnist_images']):
            mnist_img_pil = Image.fromarray(mnist_img)
            filename = os.path.join(save_dir, f"mnist_char_{pdf_basename}_{timestamp}_{i+1}.png")
            mnist_img_pil.save(filename)
        messagebox.showinfo("Download Complete", f"MNIST images saved successfully to:\n{save_dir}")
    except Exception as e:
        messagebox.showerror("Download Error", f"Failed to save MNIST images: {e}")

def show_debug_window(file_path=None):
    """Show a single debug window with navigation for all processed PDFs."""
    global debug_window, debug_canvas, debug_scrollable_frame
    
    if not debug_images:
        messagebox.showinfo("Debug Info", "No debug information available.")
        return

    if 'debug_window' in globals() and debug_window.winfo_exists():
        debug_window.destroy()

    debug_window = tk.Toplevel(root)
    debug_window.title("Debug Details")
    debug_window.geometry("800x1000")

    main_frame = tk.Frame(debug_window)
    main_frame.pack(fill=tk.BOTH, expand=True)

    control_frame = tk.Frame(main_frame)
    control_frame.pack(fill=tk.X, pady=5)
    
    pdf_label = tk.Label(control_frame, text="Select PDF:")
    pdf_label.pack(side=tk.LEFT, padx=5)
    
    pdf_combobox = ttk.Combobox(control_frame, width=40, state="readonly")
    pdf_combobox.pack(side=tk.LEFT, padx=5)
    pdf_combobox['values'] = [os.path.basename(path) for path in debug_images.keys()]
    
    if file_path and file_path in debug_images:
        pdf_combobox.current(list(debug_images.keys()).index(file_path))
    elif debug_images:
        pdf_combobox.current(0)

    canvas_frame = tk.Frame(main_frame)
    canvas_frame.pack(fill=tk.BOTH, expand=True)
    
    debug_canvas = tk.Canvas(canvas_frame)
    scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=debug_canvas.yview)
    
    debug_scrollable_frame = tk.Frame(debug_canvas)
    debug_scrollable_frame.bind(
        "<Configure>",
        lambda e: debug_canvas.configure(scrollregion=debug_canvas.bbox("all"))
    )
    
    debug_canvas.create_window((0, 0), window=debug_scrollable_frame, anchor="nw")
    debug_canvas.configure(yscrollcommand=scrollbar.set)
    
    debug_canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    def update_debug_display(event=None):
        """Update the debug window content based on selected PDF."""
        for widget in debug_scrollable_frame.winfo_children():
            widget.destroy()
            
        selected_index = pdf_combobox.current()
        if selected_index < 0:
            return
            
        selected_file = list(debug_images.keys())[selected_index]
        debug_data = debug_images[selected_file]
        row = 0

        # Cropped Image Section
        if 'cropped' in debug_data and debug_data['cropped']:
            cropped_frame = tk.Frame(debug_scrollable_frame)
            cropped_frame.grid(row=row, column=0, sticky="w", padx=10, pady=5)
            
            tk.Label(
                cropped_frame, 
                text="Cropped Image:", 
                font=("Arial", 12)
            ).pack(side=tk.LEFT, padx=5)
            
            tk.Button(
                cropped_frame, 
                text="Download", 
                command=lambda: download_image(debug_data['cropped'], "cropped", debug_window, selected_file),
                width=10,
                height=1,
                bg="#ffffff",
                fg="black",
                font=("Arial", 10, "bold")
            ).pack(side=tk.LEFT, padx=5)
            
            row += 1
            
            img = debug_data['cropped'].copy()
            img.thumbnail((400, 400))
            tk_img = ImageTk.PhotoImage(img)
            label = tk.Label(debug_scrollable_frame, image=tk_img)
            label.image = tk_img
            label.grid(row=row, column=0, padx=10, pady=5, sticky="w")
            row += 1

        # Intermediate Images Section
        if 'intermediate' in debug_data and debug_data['intermediate']:
            intermediate_steps = [
                ('grayscale', "Grayscale Image"),
                ('enhanced_grayscale', "Enhanced Grayscale Image"),  # Add enhanced grayscale
                ('binary', "Binary Threshold Image"),
                ('line_removed', "Line Removed Image"),
            ]
            
            for key, title in intermediate_steps:
                if key in debug_data['intermediate'] and debug_data['intermediate'][key]:
                    frame = tk.Frame(debug_scrollable_frame)
                    frame.grid(row=row, column=0, sticky="w", padx=10, pady=5)
                    
                    tk.Label(
                        frame, 
                        text=f"{title}:", 
                        font=("Arial", 12)
                    ).pack(side=tk.LEFT, padx=5)
                    
                    tk.Button(
                        frame, 
                        text="Download", 
                        command=lambda k=key: download_image(debug_data['intermediate'][k], k, debug_window, selected_file),
                        width=10,
                        height=1,
                        bg="#ffffff",
                        fg="black",
                        font=("Arial", 10, "bold")
                    ).pack(side=tk.LEFT, padx=5)
                    
                    row += 1
                    
                    img = debug_data['intermediate'][key].copy()
                    img.thumbnail((400, 400))
                    tk_img = ImageTk.PhotoImage(img)
                    label = tk.Label(debug_scrollable_frame, image=tk_img)
                    label.image = tk_img
                    label.grid(row=row, column=0, padx=10, pady=5, sticky="w")
                    row += 1

        # Processed Image Section
        if 'processed' in debug_data and debug_data['processed']:
            processed_frame = tk.Frame(debug_scrollable_frame)
            processed_frame.grid(row=row, column=0, sticky="w", padx=10, pady=5)
            
            tk.Label(
                processed_frame, 
                text="Processed Image (Noise Cleaned):", 
                font=("Arial", 12)
            ).pack(side=tk.LEFT, padx=5)
            
            tk.Button(
                processed_frame, 
                text="Download", 
                command=lambda: download_image(debug_data['processed'], "processed", debug_window, selected_file),
                width=10,
                height=1,
                bg="#ffffff",
                fg="black",
                font=("Arial", 10, "bold")
            ).pack(side=tk.LEFT, padx=5)
            
            row += 1
            
            img = debug_data['processed'].copy()
            img.thumbnail((400, 400))
            tk_img = ImageTk.PhotoImage(img)
            label = tk.Label(debug_scrollable_frame, image=tk_img)
            label.image = tk_img
            label.grid(row=row, column=0, padx=10, pady=5, sticky="w")
            row += 1
        
        # Digit Detection Section
        if 'green_boxes_img' in debug_data and debug_data['green_boxes_img']:
            detection_frame = tk.Frame(debug_scrollable_frame)
            detection_frame.grid(row=row, column=0, sticky="w", padx=10, pady=5)
            
            tk.Label(
                detection_frame, 
                text="Handwritten Characters Detection (Green Bounding Boxes):", 
                font=("Arial", 12)
            ).pack(side=tk.LEFT, padx=5)
            
            tk.Button(
                detection_frame, 
                text="Download", 
                command=lambda: download_image(debug_data['green_boxes_img'], "digit_detection", debug_window, selected_file),
                width=10,
                height=1,
                bg="#ffffff",
                fg="black",
                font=("Arial", 10, "bold")
            ).pack(side=tk.LEFT, padx=5)
            
            row += 1
            
            img = debug_data['green_boxes_img'].copy()
            img.thumbnail((400, 400))
            tk_img = ImageTk.PhotoImage(img)
            label = tk.Label(debug_scrollable_frame, image=tk_img)
            label.image = tk_img
            label.grid(row=row, column=0, padx=10, pady=5, sticky="w")
            row += 1
        
        # MNIST Images Section
        if 'mnist_images' in debug_data and debug_data['mnist_images']:
            mnist_title_frame = tk.Frame(debug_scrollable_frame)
            mnist_title_frame.grid(row=row, column=0, sticky="w", padx=10, pady=5)
            
            tk.Label(
                mnist_title_frame, 
                text="MNIST-Formatted Images:", 
                font=("Arial", 12)
            ).pack(side=tk.LEFT, padx=5)
            
            tk.Button(
                mnist_title_frame, 
                text="Download", 
                command=lambda: download_mnist_images(debug_data, debug_window, selected_file),
                width=10,
                height=1,
                bg="#ffffff",
                fg="black",
                font=("Arial", 10, "bold")
            ).pack(side=tk.LEFT, padx=5)
            
            row += 1
            
            mnist_frame = tk.Frame(debug_scrollable_frame)
            mnist_frame.grid(row=row, column=0, padx=10, pady=5, sticky="w")
            
            for i, mnist_img in enumerate(debug_data['mnist_images']):
                mnist_img_pil = Image.fromarray(mnist_img)
                mnist_img_pil = mnist_img_pil.resize((80, 80), Image.NEAREST)
                tk_img = ImageTk.PhotoImage(mnist_img_pil)
                
                digit_frame = tk.Frame(mnist_frame)
                digit_frame.pack(side=tk.LEFT, padx=5, pady=5)
                
                label = tk.Label(digit_frame, image=tk_img)
                label.image = tk_img
                label.pack()
            
            row += 1

        tk.Button(
            debug_scrollable_frame, 
            text="Close", 
            command=debug_window.destroy, 
            width=20, 
            height=2
        ).grid(row=row, column=0, pady=20)

    pdf_combobox.bind("<<ComboboxSelected>>", update_debug_display)
    update_debug_display()

def setup_canvas():
    """Configure the canvas when the window is resized."""
    if pdf_image:
        display_image(pdf_image)

def on_combobox_select(event):
    """Handle selection change in the PDF combobox."""
    display_selected_pdf()

# Initialize the main application
root = tk.Tk()
root.title("PDF Image Processor")
root.state('zoomed')

try:
    style = ttk.Style()
    available_themes = style.theme_names()
    if 'clam' in available_themes:
        style.theme_use('clam')
    elif 'vista' in available_themes:
        style.theme_use('vista')
except Exception:
    pass

# Global variables
pdf_image = None
tk_image = None
scaled_image = None
image_scale = 1
start_x = start_y = 0
rect_id = None
crop_coords = None
pdf_files = []
debug_images = {}
canvas_offset_x = canvas_offset_y = 0
original_image = None

main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

control_frame = tk.Frame(main_frame)
control_frame.pack(fill=tk.X, pady=10)

upload_button = tk.Button(control_frame, text="Upload PDFs", command=load_pdf, width=15, height=2)
upload_button.grid(row=0, column=0, padx=5)

pdf_label = tk.Label(control_frame, text="Select PDF:")
pdf_label.grid(row=0, column=1, padx=5, pady=5, sticky="e")

pdf_combobox = ttk.Combobox(control_frame, width=40, state="readonly")
pdf_combobox.grid(row=0, column=2, padx=5, pady=5, sticky="w")
pdf_combobox.bind("<<ComboboxSelected>>", on_combobox_select)

process_button = tk.Button(
    control_frame, 
    text="Process", 
    command=process,
    width=15,
    height=2,
    bg="#4CAF50",
    fg="black",
    font=("Arial", 10, "bold")
)
process_button.grid(row=0, column=3, padx=5, pady=5)

status_label = tk.Label(
    control_frame, 
    text="Ready. Please load a PDF file.", 
    font=("Arial", 10),
    bd=1,
    relief=tk.SUNKEN,
    anchor=tk.W
)
status_label.grid(row=0, column=4, columnspan=1, padx=5, pady=5, sticky="ew")

canvas_frame = tk.Frame(main_frame, bd=2, relief=tk.SUNKEN)
canvas_frame.pack(fill=tk.BOTH, expand=True, pady=10)

canvas = tk.Canvas(canvas_frame, bg="light gray")
canvas.pack(fill=tk.BOTH, expand=True)

instructions = tk.Label(
    main_frame, 
    text="Instructions: Upload a PDF, select an area by drawing a rectangle, then click 'Process'",
    font=("Arial", 10),
    fg="#666666"
)
instructions.pack(pady=5, anchor=tk.W)

canvas.bind("<ButtonPress-1>", start_crop)
canvas.bind("<B1-Motion>", update_crop)
canvas.bind("<ButtonRelease-1>", finish_crop)
canvas.bind("<Configure>", lambda e: setup_canvas())

root.mainloop()