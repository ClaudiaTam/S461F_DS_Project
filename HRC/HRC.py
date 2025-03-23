import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from PIL import Image, ImageTk
from pdf2image import convert_from_path
import torch
from torchvision import transforms
import cv2
import numpy as np
import time
import os

# Define the digit classification model architecture (10 classes: 0-9)
class DigitNet(torch.nn.Module):
    def __init__(self):
        super(DigitNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, 3, 1)
        self.conv2 = torch.nn.Conv2d(64, 128, 3, 1)
        self.conv3 = torch.nn.Conv2d(128, 256, 3, 1)
        self.dropout1 = torch.nn.Dropout(0.25)
        self.dropout2 = torch.nn.Dropout(0.4)
        self.fc1 = torch.nn.Linear(256 * 5 * 5, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.conv3(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x = self.fc3(x)
        output = torch.nn.functional.log_softmax(x, dim=1)
        return output

# Define the binary classification model architecture (2 classes: digit or letter)
class BinaryNet(torch.nn.Module):
    def __init__(self):
        super(BinaryNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, 3, 1)
        self.conv2 = torch.nn.Conv2d(64, 128, 3, 1)
        self.conv3 = torch.nn.Conv2d(128, 256, 3, 1)
        self.dropout1 = torch.nn.Dropout(0.25)
        self.dropout2 = torch.nn.Dropout(0.4)
        self.fc1 = torch.nn.Linear(256 * 5 * 5, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 2)  # 2 outputs: digit (0) or letter (1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.conv3(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x = self.fc3(x)
        output = torch.nn.functional.log_softmax(x, dim=1)
        return output

# Load the pre-trained digit classification model
digit_model = DigitNet()
digit_model_file = "/Users/ryan/Desktop/DL/FYP/Combined_Model_seed_14598/combined_cnn_epoch:9_test-accuracy:99.5481_test-loss:0.0128.pt"
try:
    digit_model.load_state_dict(torch.load(digit_model_file, map_location=torch.device('cpu')))
    digit_model.eval()
except Exception as e:
    print(f"Error loading digit model: {e}")
    exit()

# Load the pre-trained binary classification model
binary_model = BinaryNet()
binary_model_file = "/Users/ryan/Desktop/DL/FYP/Binary_Model_seed_67675/binary_cnn_epoch:54_test-accuracy:99.9306_test-loss:0.0046.pt"  # Replace with your actual binary model path
try:
    binary_model.load_state_dict(torch.load(binary_model_file, map_location=torch.device('cpu')))
    binary_model.eval()
except Exception as e:
    print(f"Error loading binary model: {e}")
    exit()

# Image transformation for MNIST compatibility
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def classify_image(image_path):
    """Classify an image with both digit and binary models."""
    try:
        image = Image.open(image_path).convert("L")
        image_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            # Digit classification
            digit_output = digit_model(image_tensor)
            digit_pred = digit_output.argmax(dim=1, keepdim=True).item()
            # Binary classification
            binary_output = binary_model(image_tensor)
            binary_pred = binary_output.argmax(dim=1, keepdim=True).item()
            binary_label = "Digit" if binary_pred == 0 else "Letter"
        return digit_pred, binary_label
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None

def load_pdf():
    """Load and display multiple one-page PDF files."""
    global pdf_files, pdf_image, debug_images
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
    global canvas, tk_image, scaled_image, image_scale, canvas_offset_x, canvas_offset_y
    canvas.delete("all")
    image_width, image_height = image.size
    canvas_width, canvas_height = canvas.winfo_width(), canvas.winfo_height()
    if canvas_width <= 1:  # Canvas not yet drawn
        canvas_width, canvas_height = 600, 800
    scale_factor = min(canvas_width / image_width, canvas_height / image_height)
    new_width = int(image_width * scale_factor)
    new_height = int(image_height * scale_factor)
    scaled_image = image.resize((new_width, new_height))
    image_scale = scale_factor
    canvas_offset_x = (canvas_width - new_width) // 2
    canvas_offset_y = (canvas_height - new_height) // 2
    tk_image = ImageTk.PhotoImage(scaled_image)
    canvas.create_image(canvas_offset_x, canvas_offset_y, anchor="nw", image=tk_image)

def start_crop(event):
    global start_x, start_y, rect_id
    if rect_id is not None:
        canvas.delete(rect_id)
        rect_id = None
    start_x, start_y = event.x - canvas_offset_x, event.y - canvas_offset_y
    rect_id = canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="red", width=2)

def update_crop(event):
    global rect_id
    end_x = max(min(event.x, canvas.winfo_width()), 0)
    end_y = max(min(event.y, canvas.winfo_height()), 0)
    canvas.coords(rect_id, start_x + canvas_offset_x, start_y + canvas_offset_y, end_x, end_y)

def finish_crop(event):
    global crop_coords, rect_id
    if pdf_image is None:
        messagebox.showerror("Error", "No PDF image loaded.")
        return

    try:
        end_x, end_y = event.x - canvas_offset_x, event.y - canvas_offset_y
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
        status_label.config(text="Selection complete. Ready to classify.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to calculate crop coordinates: {e}")

def process_image(image_path):
    """Process the image, detect characters, crop to MNIST format, and classify them."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(
        enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 12
    )

    # Remove vertical and horizontal lines
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

    # Clean up noise
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned_binary = cv2.morphologyEx(cleaned_binary, cv2.MORPH_CLOSE, closing_kernel)
    cleaned_binary = cv2.medianBlur(cleaned_binary, 3)
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    repaired_binary = cv2.dilate(cleaned_binary, repair_kernel, iterations=1)

    # Additional dilation to connect disjointed parts
    connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    connected_binary = cv2.dilate(repaired_binary, connect_kernel, iterations=2)

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
    contours, _ = cv2.findContours(connected_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_dir = "/Users/ryan/Desktop/cropped_digits"
    os.makedirs(output_dir, exist_ok=True)

    img_height, img_width = repaired_binary.shape

    def merge_nearby_contours(contours, distance_threshold=10):
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
        if 2 < w < 100 and 17 < h < 100:
            if (x > margin and y > margin and 
                x + w < img_width - margin and 
                y + h < img_height - margin):
                initial_boxes.append(contour)

    bounding_boxes = merge_nearby_contours(initial_boxes, distance_threshold=10)
    bounding_boxes.sort(key=lambda box: box[0])

    digit_results = []
    binary_results = []
    mnist_images = []
    green_boxes_img = cv2.cvtColor(cleaned_binary.copy(), cv2.COLOR_GRAY2BGR)

    for i, (x, y, w, h) in enumerate(bounding_boxes):
        digit_roi = cleaned_binary[y:y+h, x:x+w]
        mnist_digit = preprocess_to_mnist_format(digit_roi)
        mnist_images.append(mnist_digit)
        digit_filename = os.path.join(output_dir, f"char_{i+1}_mnist.png")
        cv2.imwrite(digit_filename, mnist_digit)
        digit_pred, binary_label = classify_image(digit_filename)
        if digit_pred is not None:
            digit_results.append(str(digit_pred))
            binary_results.append(binary_label)
        cv2.rectangle(green_boxes_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        os.remove(digit_filename)

    return digit_results, cleaned_binary, mnist_images, green_boxes_img, binary_results

def classify():
    """Classify the cropped region of the PDF image and store debug images."""
    global crop_coords, pdf_files, rect_id, replace_var, user_entry_var, debug_images
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

    recognition_results = []
    file_paths = pdf_files
    
    for file_path in file_paths:
        try:
            images = convert_from_path(file_path, dpi=200)
            if len(images) != 1:
                recognition_results.append(f"{os.path.basename(file_path)}: Not a one-page PDF.")
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
                'binary_results': None  # Add binary results to debug entry
            }
            
            digit_results, processed_img, mnist_images, green_boxes_img, binary_results = process_image(temp_image_path)
            processed_img_pil = Image.fromarray(processed_img)
            green_boxes_img_pil = Image.fromarray(green_boxes_img)

            debug_entry.update({
                'processed': processed_img_pil,
                'mnist_images': mnist_images,
                'results': ''.join(digit_results),
                'green_boxes_img': green_boxes_img_pil,
                'binary_results': binary_results
            })
            
            recognition_results.append(f"{os.path.basename(file_path)}: {''.join(digit_results)}")
            
            debug_images[file_path] = debug_entry
            
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
        except Exception as e:
            recognition_results.append(f"{os.path.basename(file_path)}: Error processing file - {e}")

    if replace_var.get() and user_entry_var.get().strip():
        user_input = user_entry_var.get().strip()
        modified_results = []
        for result in recognition_results:
            parts = result.split(': ', 1)
            if len(parts) == 2:
                filename, content = parts
                if content.isdigit():
                    content_list = list(content)
                    replace_length = min(len(user_input), len(content_list))
                    for i in range(replace_length):
                        content_list[i] = user_input[i]
                    new_content = ''.join(content_list)
                    modified_results.append(f"{filename}: {new_content}")
                else:
                    modified_results.append(result)
            else:
                modified_results.append(result)
        recognition_results = modified_results

        for file_path in file_paths:
            if file_path in debug_images:
                file_index = file_paths.index(file_path)
                if 0 <= file_index < len(recognition_results):
                    debug_images[file_path]['results'] = recognition_results[file_index].split(': ')[1]

    status_label.config(text="Classification complete!")
    show_recognition_results(recognition_results)

def show_recognition_results(results):
    """Display recognition results in a scrollable text window."""
    result_window = tk.Toplevel(root)
    result_window.title("Recognition Results")
    result_window.geometry("600x400")
    
    control_frame = tk.Frame(result_window)
    control_frame.pack(fill=tk.X, padx=10, pady=5)
    
    pdf_label = tk.Label(control_frame, text="Select PDF for Debug:")
    pdf_label.pack(side=tk.LEFT, padx=5)
    
    pdf_debug_combobox = ttk.Combobox(control_frame, width=40, state="readonly")
    pdf_debug_combobox.pack(side=tk.LEFT, padx=5)
    pdf_debug_combobox['values'] = [os.path.basename(path) for path in pdf_files]
    if pdf_files:
        pdf_debug_combobox.current(pdf_combobox.current())
    
    def on_debug_select():
        selected_index = pdf_debug_combobox.current()
        if selected_index >= 0:
            show_debug_window(pdf_files[selected_index])
    
    debug_btn = tk.Button(control_frame, text="Show Debug Details", command=on_debug_select)
    debug_btn.pack(side=tk.LEFT, padx=5)
    
    close_btn = tk.Button(control_frame, text="Close", command=result_window.destroy)
    close_btn.pack(side=tk.RIGHT, padx=5)
    
    text_area = scrolledtext.ScrolledText(result_window, wrap=tk.WORD, width=60, height=20, font=("Arial", 14))
    text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    for result in results:
        text_area.insert(tk.END, result + "\n")
    text_area.configure(state="disabled")

def download_mnist_images(debug_data, parent_window, file_path=None):
    """Prompt user to save MNIST-formatted images with unique filenames."""
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
    """Show a debug window with processing details including binary classification."""
    if not file_path or file_path not in debug_images:
        messagebox.showinfo("Debug Info", "No debug information available.")
        return
        
    debug_window = tk.Toplevel(root)
    debug_window.title(f"Debug Details - {os.path.basename(file_path)}")
    debug_window.geometry("800x600")
    
    main_frame = tk.Frame(debug_window)
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    canvas = tk.Canvas(main_frame)
    scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
    
    scrollable_frame = tk.Frame(canvas)
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    debug_data = debug_images[file_path]
    row = 0
    
    if 'results' in debug_data:
        tk.Label(
            scrollable_frame, 
            text=f"Recognition Result: {debug_data['results']}", 
            font=("Arial", 14, "bold")
        ).grid(row=row, column=0, columnspan=2, pady=10, sticky="w", padx=10)
        row += 1
    
    if 'cropped' in debug_data and debug_data['cropped']:
        tk.Label(scrollable_frame, text="Cropped Image:", font=("Arial", 12)).grid(row=row, column=0, sticky="w", padx=10, pady=5)
        row += 1
        img = debug_data['cropped'].copy()
        img.thumbnail((400, 400))
        tk_img = ImageTk.PhotoImage(img)
        label = tk.Label(scrollable_frame, image=tk_img)
        label.image = tk_img
        label.grid(row=row, column=0, padx=10, pady=5, sticky="w")
        row += 1
    
    if 'processed' in debug_data and debug_data['processed']:
        tk.Label(scrollable_frame, text="Processed Image (After Preprocessing):", font=("Arial", 12)).grid(row=row, column=0, sticky="w", padx=10, pady=5)
        row += 1
        img = debug_data['processed'].copy()
        img.thumbnail((400, 400))
        tk_img = ImageTk.PhotoImage(img)
        label = tk.Label(scrollable_frame, image=tk_img)
        label.image = tk_img
        label.grid(row=row, column=0, padx=10, pady=5, sticky="w")
        row += 1
    
    if 'green_boxes_img' in debug_data and debug_data['green_boxes_img']:
        tk.Label(scrollable_frame, text="Digit Detection (Green Bounding Boxes):", font=("Arial", 12)).grid(row=row, column=0, sticky="w", padx=10, pady=5)
        row += 1
        img = debug_data['green_boxes_img'].copy()
        img.thumbnail((400, 400))
        tk_img = ImageTk.PhotoImage(img)
        label = tk.Label(scrollable_frame, image=tk_img)
        label.image = tk_img
        label.grid(row=row, column=0, padx=10, pady=5, sticky="w")
        row += 1
    
    if 'mnist_images' in debug_data and debug_data['mnist_images']:
        mnist_title_frame = tk.Frame(scrollable_frame)
        mnist_title_frame.grid(row=row, column=0, sticky="w", padx=10, pady=5)
        
        tk.Label(
            mnist_title_frame, 
            text="MNIST-Formatted Images (Used for Classification):", 
            font=("Arial", 12)
        ).pack(side=tk.LEFT, padx=5)
        
        download_btn = tk.Button(
            mnist_title_frame, 
            text="Download", 
            command=lambda: download_mnist_images(debug_data, debug_window, file_path),
            width=10,
            height=1,
            bg="#ffffff",
            fg="black",
            font=("Arial", 10, "bold")
        )
        download_btn.pack(side=tk.LEFT, padx=5)
        
        row += 1
        
        mnist_frame = tk.Frame(scrollable_frame)
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
            
            # Display binary classification result above digit result
            if 'binary_results' in debug_data and i < len(debug_data['binary_results']):
                tk.Label(
                    digit_frame, 
                    text=f"{debug_data['binary_results'][i]}", 
                    font=("Arial", 10, "bold"),
                    fg="blue"  # Optional: color to distinguish binary result
                ).pack()
            if 'results' in debug_data and i < len(debug_data['results']):
                tk.Label(
                    digit_frame, 
                    text=f"{debug_data['results'][i]}", 
                    font=("Arial", 10, "bold")
                ).pack()
        
        row += 1

    tk.Button(scrollable_frame, text="Close", command=debug_window.destroy, width=20, height=2).grid(row=row, column=0, pady=20)

def setup_canvas():
    """Configure the canvas when the window is resized."""
    if pdf_image:
        display_image(pdf_image)

def on_combobox_select(event):
    """Handle selection change in the PDF combobox."""
    display_selected_pdf()

# Initialize the main application
root = tk.Tk()
root.title("New HRC - Handwritten Recognition")
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

replace_var = tk.BooleanVar(value=False)
replace_check = tk.Checkbutton(control_frame, text="Replace first digits with:", variable=replace_var)
replace_check.grid(row=1, column=0, padx=5, pady=5, sticky="e")

user_entry_var = tk.StringVar()
user_entry = tk.Entry(
    control_frame, 
    textvariable=user_entry_var, 
    width=5,
    font=("Arial", 12),
    validate="key",
    validatecommand=(root.register(lambda p: len(p) <= 4), '%P')
)
user_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")

classify_button = tk.Button(
    control_frame, 
    text="Classify", 
    command=classify,
    width=15,
    height=2,
    bg="#4CAF50",
    fg="black",
    font=("Arial", 10, "bold")
)
classify_button.grid(row=1, column=2, padx=5, pady=5)

status_label = tk.Label(
    control_frame, 
    text="Ready. Please load a PDF file.", 
    font=("Arial", 10),
    bd=1,
    relief=tk.SUNKEN,
    anchor=tk.W
)
status_label.grid(row=1, column=3, columnspan=1, padx=5, pady=5, sticky="ew")

canvas_frame = tk.Frame(main_frame, bd=2, relief=tk.SUNKEN)
canvas_frame.pack(fill=tk.BOTH, expand=True, pady=10)

canvas = tk.Canvas(canvas_frame, bg="light gray")
canvas.pack(fill=tk.BOTH, expand=True)

instructions = tk.Label(
    main_frame, 
    text="Instructions: Upload a PDF, select an area with digits by drawing a rectangle, then click 'Classify'",
    font=("Arial", 10),
    fg="#666666"
)
instructions.pack(pady=5, anchor=tk.W)

canvas.bind("<ButtonPress-1>", start_crop)
canvas.bind("<B1-Motion>", update_crop)
canvas.bind("<ButtonRelease-1>", finish_crop)
canvas.bind("<Configure>", lambda e: setup_canvas())

root.mainloop()
