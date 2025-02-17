import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
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
        # Increased number of filters in convolutional layers
        self.conv1 = torch.nn.Conv2d(1, 64, 3, 1)  # Input channels: 1, Output channels: 64
        self.conv2 = torch.nn.Conv2d(64, 128, 3, 1)  # Output channels: 128
        self.conv3 = torch.nn.Conv2d(128, 256, 3, 1)  # Added a third convolutional layer
        self.dropout1 = torch.nn.Dropout(0.25)
        self.dropout2 = torch.nn.Dropout(0.4)
        self.fc1 = torch.nn.Linear(256 * 5 * 5, 512)  # Adjusted input size for the new conv layer
        self.fc2 = torch.nn.Linear(512, 256)  # Added an additional fully connected layer
        self.fc3 = torch.nn.Linear(256, 10)  # Output layer

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.conv3(x)  # Added the third convolutional layer
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)  # Added the additional fully connected layer
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
    global pdf_files, pdf_image
    file_paths = filedialog.askopenfilenames(
        filetypes=[("PDF Files", "*.pdf")],
        title="Select one or more one-page PDF files"
    )
    if not file_paths:
        return
    pdf_files = list(file_paths)
    display_first_pdf()

def display_first_pdf():
    global pdf_image, pdf_files
    if not pdf_files:
        return
    try:
        images = convert_from_path(pdf_files[0], dpi=200)
        if len(images) != 1:
            messagebox.showerror("Error", "Please upload one-page PDF files.")
            return
        pdf_image = images[0]
        display_image(pdf_image)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load PDF: {e}")

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
    global start_x, start_y, rect_id
    start_x, start_y = event.x - canvas_offset_x, event.y - canvas_offset_y
    rect_id = canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="red", width=2)

def update_crop(event):
    global rect_id
    end_x = max(min(event.x, canvas.winfo_width()), 0)
    end_y = max(min(event.y, canvas.winfo_height()), 0)
    canvas.coords(rect_id, start_x + canvas_offset_x, start_y + canvas_offset_y, end_x, end_y)

def finish_crop(event):
    global pdf_image, scaled_image, image_scale, rect_id, pdf_files
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
        
        recognition_results = []
        for file_path in pdf_files:
            try:
                images = convert_from_path(file_path, dpi=200)
                if len(images) != 1:
                    recognition_results.append(f"{os.path.basename(file_path)}: Not a one-page PDF.")
                    continue
                
                cropped_image = images[0].crop(crop_coords)
                temp_image_path = "temp_cropped_region.png"
                cropped_image.save(temp_image_path)
                
                localized_boxes = localization(temp_image_path)
                if localized_boxes:
                    localized_boxes.sort(key=lambda b: b[0])
                    leftmost_box = localized_boxes[0]
                    rightmost_box = localized_boxes[-1]
                    final_x1, final_y1 = leftmost_box[0], leftmost_box[1]
                    final_x2, final_y2 = rightmost_box[0] + rightmost_box[2], rightmost_box[1] + rightmost_box[3]
                    final_crop_box = (final_x1, final_y1, final_x2, final_y2)
                    final_cropped_image = cropped_image.crop(final_crop_box)
                    final_image_path = "final_cropped_region.png"
                    final_cropped_image.save(final_image_path)
                    
                    results = process_image(final_image_path)
                    recognition_results.append(f"{os.path.basename(file_path)}: {''.join(map(str, results))}")
                else:
                    recognition_results.append(f"{os.path.basename(file_path)}: No regions detected.")
            except Exception as e:
                recognition_results.append(f"{os.path.basename(file_path)}: Error processing file - {e}")
        
        # Apply user replacement if enabled
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
        
        show_recognition_results(recognition_results)
    
    except Exception as e:
        messagebox.showerror("Error", f"Failed to process image: {e}")
    finally:
        if 'temp_image_path' in locals() and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        if rect_id:
            canvas.delete(rect_id)

def localization(image_path, margin=2):
    image = cv2.imread(image_path)
    INV_image = cv2.bitwise_not(image)
    gray = cv2.cvtColor(INV_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 30 < w < 200 and 30 < h < 150:
            bounding_boxes.append((x, y, w, h))
    final_boxes = []
    image_height, image_width = image.shape[:2]
    for i, (x1, y1, w1, h1) in enumerate(bounding_boxes):
        is_nested = False
        for j, (x2, y2, w2, h2) in enumerate(bounding_boxes):
            if i != j:
                if x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2:
                    is_nested = True
                    break
        if not is_nested and x1 >= margin and y1 >= margin and x1 + w1 <= image_width - margin and y1 + h1 <= image_height - margin:
            final_boxes.append((x1, y1, w1, h1))
    for (x, y, w, h) in final_boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return final_boxes

def process_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 12)
    border_size = 5  # Adjust based on noise size
    height, width = binary.shape
    binary[:border_size, :] = 0          # Clear top border
    binary[-border_size:, :] = 0         # Clear bottom border
    binary[:, :border_size] = 0          # Clear left border
    #binary[:, -border_size:] = 0         # Clear right border
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
        digit = image[y:y+h, x:x+w]
        max_dim = max(w, h)
        if max_dim == 0:
            return np.zeros((28, 28), dtype=np.uint8)
        scale = 20.0 / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((28, 28), dtype=np.uint8)
        x_offset = (28 - new_w) // 2
        y_offset = (28 - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        #vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,2))
        #dilated_canvas = cv2.dilate(canvas, vertical_kernel, iterations=1)
        return canvas

    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 5 < w < 50 and 17 < h < 60:
            bounding_boxes.append((x, y, w, h))
    bounding_boxes.sort(key=lambda box: box[0])
    results = []
    for i, (x, y, w, h) in enumerate(bounding_boxes):
        digit_roi = cleaned_binary[y:y+h, x:x+w]
        mnist_digit = preprocess_to_mnist_format(digit_roi)
        digit_filename = os.path.join(output_dir, f"digit_{i+1}_mnist.png")
        cv2.imwrite(digit_filename, mnist_digit)
        prediction = classify_image(digit_filename)
        results.append(prediction)
        cv2.rectangle(cleaned_binary, (x, y), (x + w, y + h), (0, 0, 0), 1)
        os.remove(digit_filename)
    return results

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


# Modified GUI setup
root = tk.Tk()
root.title("HRC")

# Add GUI elements for replacement functionality
replace_var = tk.BooleanVar(value=False)
user_entry_var = tk.StringVar()

def validate_length(new_text):
    return len(new_text) <= 4

pdf_image = None
tk_image = None
scaled_image = None
image_scale = 1
start_x = start_y = 0
rect_id = None
pdf_files = []

# Create main control frame
control_frame = tk.Frame(root)
control_frame.pack(pady=10)

# Upload button
upload_button = tk.Button(control_frame, text="Upload PDFs", command=load_pdf)
upload_button.grid(row=0, column=0, padx=5)

# Replacement checkbox and entry
replace_check = tk.Checkbutton(
    control_frame, 
    text="Replace first digits with:", 
    variable=replace_var
)
replace_check.grid(row=0, column=1, padx=5)

user_entry = tk.Entry(
    control_frame, 
    textvariable=user_entry_var, 
    width=5,
    validate="key",
    validatecommand=(root.register(validate_length), '%P')
)
user_entry.grid(row=0, column=2, padx=5)

canvas = tk.Canvas(root, width=600, height=800, bg="gray")
canvas.pack(pady=10)
canvas.bind("<ButtonPress-1>", start_crop)
canvas.bind("<B1-Motion>", update_crop)
canvas.bind("<ButtonRelease-1>", finish_crop)
root.state('zoomed')
root.mainloop()
