import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
from pdf2image import convert_from_path
import torch
from torchvision import transforms
import cv2
import numpy as np
import os

# Define the model architecture
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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

model = Net()
model_file = "/Users/ryan/Desktop/DL/FYP/MNIST_Model_seed_83082/mnist_cnn_epoch:6_test-accuracy:99.5000_test-loss:0.0169.pt"
try:
    model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def classify_image(image_path):
    try:
        image = Image.open(image_path).convert("L")
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(image)
            prediction = output.argmax(dim=1, keepdim=True).item()
        return prediction
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def load_pdf():
    global pdf_files, pdf_image, debug_images
    file_paths = filedialog.askopenfilenames(
        filetypes=[("PDF Files", "*.pdf")],
        title="Select one or more one-page PDF files"
    )
    if not file_paths:
        return
    pdf_files = list(file_paths)
    debug_images.clear()
    pdf_listbox.delete(0, tk.END)
    for file_path in pdf_files:
        pdf_listbox.insert(tk.END, os.path.basename(file_path))
    display_first_pdf()

def display_pdf_image(file_path):
    global pdf_image, tk_image, scaled_image, image_scale, canvas_offset_x, canvas_offset_y
    try:
        images = convert_from_path(file_path, dpi=200)
        if len(images) != 1:
            messagebox.showerror("Error", "Please upload one-page PDF files.")
            return
        pdf_image = images[0]
        display_image(pdf_image)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load PDF: {e}")

def display_first_pdf():
    if not pdf_files:
        return
    display_pdf_image(pdf_files[0])

def display_image(image):
    global canvas, tk_image, scaled_image, image_scale, canvas_offset_x, canvas_offset_y
    canvas.delete("all")
    image_width, image_height = image.size
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
    global start_x, start_y, current_rect_id
    start_x, start_y = event.x - canvas_offset_x, event.y - canvas_offset_y
    if current_rect_id:
        canvas.delete(current_rect_id)
    current_rect_id = canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="red", width=2)

def update_crop(event):
    global current_rect_id
    end_x = max(min(event.x, canvas.winfo_width()), 0)
    end_y = max(min(event.y, canvas.winfo_height()), 0)
    canvas.coords(current_rect_id, start_x + canvas_offset_x, start_y + canvas_offset_y, end_x, end_y)

def finish_crop(event):
    global crop_coords, current_rect_id
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
        x1 = int(x1 / image_scale)
        y1 = int(y1 / image_scale)
        x2 = int(x2 / image_scale)
        y2 = int(y2 / image_scale)
        crop_coords = (x1, y1, x2, y2)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to calculate crop coordinates: {e}")

def classify():
    global crop_coords, current_rect_id, pdf_files, debug_images
    if pdf_image is None:
        messagebox.showerror("Error", "No PDF image loaded.")
        return
    if crop_coords is None:
        messagebox.showerror("Error", "Please draw a bounding box first.")
        return

    try:
        recognition_results = []
        for file_path in pdf_files:
            try:
                images = convert_from_path(file_path, dpi=200)
                if len(images) != 1:
                    recognition_results.append(f"{os.path.basename(file_path)}: Not a one-page PDF.")
                    continue
                
                original_img = images[0]
                cropped_image = original_img.crop(crop_coords)
                temp_image_path = "temp_cropped_region.png"
                cropped_image.save(temp_image_path)
                
                big_boxes, localized_img = localization(temp_image_path)
                localized_img_pil = Image.fromarray(cv2.cvtColor(localized_img, cv2.COLOR_BGR2RGB))
                
                debug_entry = {
                    'original': original_img,
                    'cropped': cropped_image,
                    'localized': localized_img_pil,
                    'processed': None,
                    'final_cropped': None
                }
                
                if big_boxes:
                    x, y, w, h = big_boxes[0]
                    final_crop_box = (x, y, x + w, y + h)
                    final_cropped_image = cropped_image.crop(final_crop_box)
                    final_image_path = "final_cropped_region.png"
                    final_cropped_image.save(final_image_path)
                    
                    results, processed_img = process_image(final_image_path)
                    processed_img_pil = Image.fromarray(processed_img)
                    
                    debug_entry.update({
                        'processed': processed_img_pil,
                        'final_cropped': final_cropped_image
                    })
                    recognition_results.append(f"{os.path.basename(file_path)}: {''.join(map(str, results))}")
                else:
                    recognition_results.append(f"{os.path.basename(file_path)}: No regions detected.")
                
                debug_images[file_path] = debug_entry
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
        
        if current_rect_id:
            canvas.delete(current_rect_id)
            current_rect_id = None
        crop_coords = None
        
        show_recognition_results(recognition_results)
    
    except Exception as e:
        messagebox.showerror("Error", f"Failed to process image: {e}")
    finally:
        if 'temp_image_path' in locals() and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        if 'final_image_path' in locals() and os.path.exists(final_image_path):
            os.remove(final_image_path)

def localization(image_path, margin=2):
    image = cv2.imread(image_path)
    INV_image = cv2.bitwise_not(image)
    gray = cv2.cvtColor(INV_image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    small_boxes = []
    image_height, image_width = image.shape[:2]
    small_box_min_w, small_box_max_w = 30, 200
    small_box_min_h, small_box_max_h = 30, 150

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if small_box_min_w < w < small_box_max_w and small_box_min_h < h < small_box_max_h:
            small_boxes.append((x, y, w, h))

    final_small_boxes = []
    for i, (x1, y1, w1, h1) in enumerate(small_boxes):
        is_nested = False
        for j, (x2, y2, w2, h2) in enumerate(small_boxes):
            if i != j and (x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2):
                is_nested = True
                break
        if not is_nested and x1 >= margin and y1 >= margin and x1 + w1 <= image_width - margin and y1 + h1 <= image_height - margin:
            final_small_boxes.append((x1, y1, w1, h1))

    big_boxes = []
    debug_img = image.copy()
    if final_small_boxes:
        min_x = min(box[0] for box in final_small_boxes)
        min_y = min(box[1] for box in final_small_boxes)
        max_x = max(box[0] + box[2] for box in final_small_boxes)
        max_y = max(box[1] + box[3] for box in final_small_boxes)
        big_box = (min_x, min_y, max_x - min_x, max_y - min_y)
        big_boxes.append(big_box)
        cv2.rectangle(debug_img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
    
    return big_boxes, debug_img

def process_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 12)
    border_size = 5
    height, width = binary.shape
    binary[:border_size, :] = 0
    binary[-border_size:, :] = 0
    binary[:, :border_size] = 0

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    eroded = cv2.erode(binary, vertical_kernel, iterations=1)
    detected_vertical_lines = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, vertical_kernel)
    dilated_vertical_lines = cv2.dilate(detected_vertical_lines, vertical_kernel, iterations=1)
    cleaned_binary = cv2.subtract(binary, dilated_vertical_lines)
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned_binary = cv2.morphologyEx(cleaned_binary, cv2.MORPH_CLOSE, closing_kernel)
    cleaned_binary = cv2.medianBlur(cleaned_binary, 3)
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    repaired_binary = cv2.dilate(cleaned_binary, repair_kernel, iterations=1)
    repaired_binary = cv2.morphologyEx(repaired_binary, cv2.MORPH_CLOSE, repair_kernel)
    contours, _ = cv2.findContours(repaired_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_dir = "../cropped_digits"
    os.makedirs(output_dir, exist_ok=True)

    def preprocess_to_mnist_format(image):
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

    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 5 < w < 50 and 17 < h < 60:
            bounding_boxes.append((x, y, w, h))
    bounding_boxes.sort(key=lambda box: box[0])
    results = []
    mnist_images = []  # List to store MNIST-preprocessed images

    # Create a copy of the cleaned_binary image to draw green bounding boxes
    green_boxes_img = cv2.cvtColor(cleaned_binary, cv2.COLOR_GRAY2BGR)

    for i, (x, y, w, h) in enumerate(bounding_boxes):
        digit_roi = cleaned_binary[y:y+h, x:x+w]
        mnist_digit = preprocess_to_mnist_format(digit_roi)
        mnist_images.append(mnist_digit)  # Save the MNIST-preprocessed image
        digit_filename = os.path.join(output_dir, f"digit_{i+1}_mnist.png")
        cv2.imwrite(digit_filename, mnist_digit)
        prediction = classify_image(digit_filename)
        results.append(prediction)
        cv2.rectangle(cleaned_binary, (x, y), (x + w, y + h), (0, 0, 0), 1)
        cv2.rectangle(green_boxes_img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green bounding boxes
        os.remove(digit_filename)
    return results, cleaned_binary, mnist_images, green_boxes_img  # Return the green_boxes_img

def show_recognition_results(results):
    result_window = tk.Toplevel(root)
    result_window.title("Recognition Results")
    text_area = scrolledtext.ScrolledText(
        result_window,
        wrap=tk.WORD,
        width=60,
        height=20,
        font=("Arial", 24)
    )
    text_area.pack(padx=10, pady=10)
    for result in results:
        text_area.insert(tk.END, result + "\n")
    text_area.configure(state="disabled")

def on_pdf_select(event):
    selected_indices = pdf_listbox.curselection()
    if not selected_indices:
        return
    selected_index = selected_indices[0]
    selected_file = pdf_files[selected_index]
    
    # Clear previous images
    for widget in image_display_frame.winfo_children():
        widget.destroy()
    
    # Display PDF image in main canvas
    display_pdf_image(selected_file)
    
    # Show debug images if available
    if selected_file not in debug_images:
        tk.Label(image_display_frame, text="No debug images available").grid(row=0, column=0)
        return
    
    debug_data = debug_images[selected_file]
    row = 0
    
    # Display recognition result
    if 'results' in debug_data:
        result_label = tk.Label(image_display_frame, text=f"Recognition Result: {debug_data['results']}", font=("Arial", 14))
        result_label.grid(row=row, column=0, columnspan=2, pady=10)
        row += 1
    
    # Cropped Image
    if 'cropped' in debug_data and debug_data['cropped']:
        img = debug_data['cropped'].copy()
        img.thumbnail((200, 200))
        tk_img = ImageTk.PhotoImage(img)
        label = tk.Label(image_display_frame, image=tk_img)
        label.image = tk_img
        label.grid(row=row, column=0, padx=5, pady=5)
        tk.Label(image_display_frame, text="Cropped").grid(row=row, column=1)
        row += 1
    
    # Localized Image
    if 'localized' in debug_data and debug_data['localized']:
        img = debug_data['localized'].copy()
        img.thumbnail((200, 200))
        tk_img = ImageTk.PhotoImage(img)
        label = tk.Label(image_display_frame, image=tk_img)
        label.image = tk_img
        label.grid(row=row, column=0, padx=5, pady=5)
        tk.Label(image_display_frame, text="Localized").grid(row=row, column=1)
        row += 1
    
    # Processed Image
    if 'processed' in debug_data and debug_data['processed']:
        img = debug_data['processed'].copy()
        img.thumbnail((200, 200))
        tk_img = ImageTk.PhotoImage(img)
        label = tk.Label(image_display_frame, image=tk_img)
        label.image = tk_img
        label.grid(row=row, column=0, padx=5, pady=5)
        tk.Label(image_display_frame, text="Processed").grid(row=row, column=1)
        row += 1
    
    # Green Bounding Boxes Image
    if 'green_boxes_img' in debug_data and debug_data['green_boxes_img']:
        img = debug_data['green_boxes_img'].copy()
        img.thumbnail((200, 200))
        tk_img = ImageTk.PhotoImage(img)
        label = tk.Label(image_display_frame, image=tk_img)
        label.image = tk_img
        label.grid(row=row, column=0, padx=5, pady=5)
        tk.Label(image_display_frame, text="Green Bounding Boxes").grid(row=row, column=1)
        row += 1
    
    # MNIST-preprocessed Images
    if 'mnist_images' in debug_data and debug_data['mnist_images']:
        mnist_label = tk.Label(image_display_frame, text="MNIST-preprocessed Images", font=("Arial", 12))
        mnist_label.grid(row=row, column=0, columnspan=2, pady=10)
        row += 1
        
        # Create a frame for MNIST images to display them horizontally
        mnist_frame = tk.Frame(image_display_frame)
        mnist_frame.grid(row=row, column=0, columnspan=2, pady=5)
        
        for i, mnist_img in enumerate(debug_data['mnist_images']):
            mnist_img_pil = Image.fromarray(mnist_img)
            mnist_img_pil.thumbnail((100, 100))  # Smaller size for MNIST images
            tk_img = ImageTk.PhotoImage(mnist_img_pil)
            label = tk.Label(mnist_frame, image=tk_img)
            label.image = tk_img
            label.pack(side=tk.LEFT, padx=5, pady=5)
            tk.Label(mnist_frame, text=f"Digit {i+1}").pack(side=tk.LEFT, padx=5, pady=5)

# Update the classify function to include MNIST-preprocessed images in debug data
def classify():
    global crop_coords, current_rect_id, pdf_files, debug_images
    if pdf_image is None:
        messagebox.showerror("Error", "No PDF image loaded.")
        return
    if crop_coords is None:
        messagebox.showerror("Error", "Please draw a bounding box first.")
        return

    try:
        recognition_results = []
        for file_path in pdf_files:
            try:
                images = convert_from_path(file_path, dpi=200)
                if len(images) != 1:
                    recognition_results.append(f"{os.path.basename(file_path)}: Not a one-page PDF.")
                    continue
                
                original_img = images[0]
                cropped_image = original_img.crop(crop_coords)
                temp_image_path = "temp_cropped_region.png"
                cropped_image.save(temp_image_path)
                
                big_boxes, localized_img = localization(temp_image_path)
                localized_img_pil = Image.fromarray(cv2.cvtColor(localized_img, cv2.COLOR_BGR2RGB))
                
                debug_entry = {
                    'original': original_img,
                    'cropped': cropped_image,
                    'localized': localized_img_pil,
                    'processed': None,
                    'final_cropped': None,
                    'mnist_images': None,  # Initialize MNIST images
                    'green_boxes_img': None  # Initialize green boxes image
                }
                
                if big_boxes:
                    x, y, w, h = big_boxes[0]
                    final_crop_box = (x, y, x + w, y + h)
                    final_cropped_image = cropped_image.crop(final_crop_box)
                    final_image_path = "final_cropped_region.png"
                    final_cropped_image.save(final_image_path)
                    
                    results, processed_img, mnist_images, green_boxes_img = process_image(final_image_path)  # Get MNIST images and green boxes image
                    processed_img_pil = Image.fromarray(processed_img)
                    green_boxes_img_pil = Image.fromarray(green_boxes_img)

                    debug_entry.update({
                        'processed': processed_img_pil,
                        'final_cropped': final_cropped_image,
                        'mnist_images': mnist_images,
                        'results': ''.join(map(str, results)),  # Store the recognition result
                        'green_boxes_img': green_boxes_img_pil  # Store the green boxes image
                    })
                    
                    recognition_results.append(f"{os.path.basename(file_path)}: {''.join(map(str, results))}")
                else:
                    recognition_results.append(f"{os.path.basename(file_path)}: No regions detected.")
                
                debug_images[file_path] = debug_entry
            except Exception as e:
                recognition_results.append(f"{os.path.basename(file_path)}: Error processing file - {e}")
        
        # Apply replacement if enabled
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

            # Update the debug_data with the modified results
            for file_path in pdf_files:
                if file_path in debug_images:
                    debug_images[file_path]['results'] = recognition_results[pdf_files.index(file_path)].split(': ')[1]
        
        if current_rect_id:
            canvas.delete(current_rect_id)
            current_rect_id = None
        crop_coords = None
        
        show_recognition_results(recognition_results)
    
    except Exception as e:
        messagebox.showerror("Error", f"Failed to process image: {e}")
    finally:
        if 'temp_image_path' in locals() and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        if 'final_image_path' in locals() and os.path.exists(final_image_path):
            os.remove(final_image_path)

root = tk.Tk()
root.title("HRC")

is_debug_visible = False  # Track sidebar visibility state

def toggle_debug():
    global is_debug_visible
    if is_debug_visible:
        right_frame.grid_remove()
        root.grid_columnconfigure(1, weight=0)
        toggle_debug_btn.config(text="Show Detail")
        is_debug_visible = False
    else:
        right_frame.grid()
        root.grid_columnconfigure(1, weight=1)
        toggle_debug_btn.config(text="Hide Detail")
        is_debug_visible = True

# Initialize variables
pdf_image = None
tk_image = None
scaled_image = None
image_scale = 1
start_x = start_y = 0
crop_coords = None
current_rect_id = None
pdf_files = []
debug_images = {}

# Create frames
left_frame = tk.Frame(root)
left_frame.grid(row=0, column=0, sticky="nsew")

right_frame = tk.Frame(root)
right_frame.grid(row=0, column=1, sticky="nsew")
right_frame.grid_remove()  # Explicitly hide the right_frame at startup

root.grid_columnconfigure(0, weight=3)
root.grid_columnconfigure(1, weight=1)
root.grid_rowconfigure(0, weight=1)

# Left frame components
control_frame = tk.Frame(left_frame)
control_frame.pack(pady=10)

replace_var = tk.BooleanVar(value=False)
user_entry_var = tk.StringVar()

upload_button = tk.Button(control_frame, text="Upload PDFs", command=load_pdf)
upload_button.grid(row=0, column=0, padx=5)

classify_button = tk.Button(control_frame, text="Classify", command=classify)
classify_button.grid(row=0, column=1, padx=5)

# Add the toggle button
toggle_debug_btn = tk.Button(control_frame, text="Show Detail", command=toggle_debug)
toggle_debug_btn.grid(row=0, column=4, padx=5)  # Adjust column index as needed

replace_check = tk.Checkbutton(
    control_frame, 
    text="Replace first digits with:", 
    variable=replace_var
)
replace_check.grid(row=0, column=2, padx=5)

user_entry = tk.Entry(
    control_frame, 
    textvariable=user_entry_var, 
    width=5,
    validate="key",
    validatecommand=(root.register(lambda p: len(p) <= 4), '%P')
)
user_entry.grid(row=0, column=3, padx=5)

canvas = tk.Canvas(left_frame, width=600, height=800, bg="gray")
canvas.pack(pady=10)
canvas.bind("<ButtonPress-1>", start_crop)
canvas.bind("<B1-Motion>", update_crop)
canvas.bind("<ButtonRelease-1>", finish_crop)

# Right frame components
pdf_listbox = tk.Listbox(right_frame)
pdf_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
pdf_listbox.bind('<<ListboxSelect>>', on_pdf_select)

image_display_frame = tk.Frame(right_frame)
image_display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

root.state('zoomed')
root.mainloop()
