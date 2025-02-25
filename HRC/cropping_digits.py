import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk
from pdf2image import convert_from_path
import cv2
import numpy as np
import os

def load_pdf():
    """Load and display a one-page PDF file."""
    global pdf_image
    file_path = filedialog.askopenfilename(
        filetypes=[("PDF Files", "*.pdf")],
        title="Select a one-page PDF file"
    )
    if not file_path:
        return

    try:
        # Convert the PDF to images (one image per page)
        images = convert_from_path(file_path, dpi=200)

        # Check if the PDF has only one page
        if len(images) != 1:
            messagebox.showerror("Error", "Please upload a one-page PDF file.")
            return

        # Get the first (and only) page as an image
        pdf_image = images[0]

        # Display the image in the GUI
        display_image(pdf_image)

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load PDF: {e}")

def display_image(image):
    """Display the given PIL image in the GUI."""
    global canvas, tk_image, scaled_image, image_scale, canvas_offset_x, canvas_offset_y
    canvas.delete("all")  # Clear previous image

    # Resize the image to fit the canvas (preserve aspect ratio)
    image_width, image_height = image.size
    canvas_width, canvas_height = 600, 800

    # Calculate scale factor and offsets to center the image
    scale_factor = min(canvas_width / image_width, canvas_height / image_height)
    new_width = int(image_width * scale_factor)
    new_height = int(image_height * scale_factor)
    scaled_image = image.resize((new_width, new_height))
    image_scale = scale_factor

    # Calculate offsets to center the image on the canvas
    canvas_offset_x = (canvas_width - new_width) // 2
    canvas_offset_y = (canvas_height - new_height) // 2

    # Convert the scaled image to a format suitable for tkinter
    tk_image = ImageTk.PhotoImage(scaled_image)
    canvas.create_image(canvas_offset_x, canvas_offset_y, anchor="nw", image=tk_image)

def start_crop(event):
    """Start the crop selection with mouse click."""
    global start_x, start_y, rect_id
    start_x, start_y = event.x - canvas_offset_x, event.y - canvas_offset_y
    rect_id = canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="red", width=2)

def update_crop(event):
    """Update the crop selection rectangle as the mouse is dragged."""
    global rect_id
    end_x = max(min(event.x, canvas.winfo_width()), 0)
    end_y = max(min(event.y, canvas.winfo_height()), 0)
    canvas.coords(rect_id, start_x + canvas_offset_x, start_y + canvas_offset_y, end_x, end_y)

def finish_crop(event):
    """Finish the crop selection and crop the image."""
    global pdf_image, scaled_image, image_scale, rect_id
    if pdf_image is None:
        messagebox.showerror("Error", "No PDF image loaded.")
        return

    try:
        # Get the rectangle coordinates on the canvas
        end_x, end_y = event.x - canvas_offset_x, event.y - canvas_offset_y
        x1, y1, x2, y2 = min(start_x, end_x), min(start_y, end_y), max(start_x, end_x), max(start_y, end_y)

        # Ensure coordinates are within the image bounds
        x1 = max(0, min(x1, scaled_image.width))
        y1 = max(0, min(y1, scaled_image.height))
        x2 = max(0, min(x2, scaled_image.width))
        y2 = max(0, min(y2, scaled_image.height))

        # Scale the coordinates back to the original image size
        x1 = int(x1 / image_scale)
        y1 = int(y1 / image_scale)
        x2 = int(x2 / image_scale)
        y2 = int(y2 / image_scale)

        # Crop the image with the scaled coordinates
        crop_box = (x1, y1, x2, y2)
        cropped_image = pdf_image.crop(crop_box)

        # Save the cropped region as a temporary image to pass to localization
        temp_image_path = "temp_cropped_region.png"
        cropped_image.save(temp_image_path)

        # Perform localization and cropping without saving cropped images
        localized_boxes = localization(temp_image_path)
        messagebox.showinfo("Success", f"Localized {len(localized_boxes)} regions. Digits saved in output directories.")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to crop image: {e}")
    finally:
        # Delete the temporary file after use
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        # Delete the rectangle after finishing the crop
        if rect_id:
            canvas.delete(rect_id)

def save_cropped_images(image, boxes, output_folder="output"):
    """Save the cropped images based on the localized boxes and process each."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, (x, y, w, h) in enumerate(boxes):
        cropped_image = image[y:y+h, x:x+w]
        output_path = os.path.join(output_folder, f"cropped_{i+1}.png")
        cv2.imwrite(output_path, cropped_image)
        print(f"Saved {output_path}")

        # Process the cropped image to extract digits
        digit_output_dir = os.path.join(output_folder, f"cropped_{i+1}_digits")
        process_image(output_path, digit_output_dir)

def localization(image_path, margin=2):
    """Localize small rectangles."""
    image = cv2.imread(image_path)
    INV_image = cv2.bitwise_not(image)
    gray = cv2.cvtColor(INV_image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    cv2.imshow("Threshold", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    small_boxes = []
    image_height, image_width = image.shape[:2]

    small_box_min_w, small_box_max_w = 30, 200
    small_box_min_h, small_box_max_h = 30, 150

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Detect small boxes
        if small_box_min_w < w < small_box_max_w and small_box_min_h < h < small_box_max_h:
            small_boxes.append((x, y, w, h))

    # Process small boxes to find encompassing rectangle
    big_boxes = []
    final_small_boxes = []
    for i, (x1, y1, w1, h1) in enumerate(small_boxes):
        is_nested = False
        for j, (x2, y2, w2, h2) in enumerate(small_boxes):
            if i != j and (x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2):
                is_nested = True
                break
        if not is_nested and x1 >= margin and y1 >= margin and x1 + w1 <= image_width - margin and y1 + h1 <= image_height - margin:
            final_small_boxes.append((x1, y1, w1, h1))

    if final_small_boxes:
        min_x = min(box[0] for box in final_small_boxes)
        min_y = min(box[1] for box in final_small_boxes)
        max_x = max(box[0] + box[2] for box in final_small_boxes)
        max_y = max(box[1] + box[3] for box in final_small_boxes)
        big_box = (min_x, min_y, max_x - min_x, max_y - min_y)
        big_boxes.append(big_box)

    # Draw bounding boxes on the visualization image
    for (x, y, w, h) in big_boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the visualization image with bounding boxes
    cv2.imshow("Localized Regions", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the cropped images from the original image (without bounding boxes)
    save_cropped_images(image, big_boxes)

    return big_boxes

def process_image(image_path, output_dir):
    """Process the image to remove vertical lines, detect words, and crop them in MNIST format."""
    # Load the image
    img = cv2.imread(image_path)

    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)

    # Apply thresholding to get a binary image (invert colors)
    binary = cv2.adaptiveThreshold(
        enhanced_gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 12
    )

    cv2.imshow("binary", binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Clear border noise
    border_size = 5
    height, width = binary.shape
    binary[:border_size, :] = 0
    binary[-border_size:, :] = 0
    binary[:, :border_size] = 0

    # Detect vertical lines using morphological operations
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    eroded = cv2.erode(binary, vertical_kernel, iterations=1)
    detected_vertical_lines = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, vertical_kernel)

    # Subtract vertical lines from binary image
    dilated_vertical_lines = cv2.dilate(detected_vertical_lines, vertical_kernel, iterations=1)
    cleaned_binary = cv2.subtract(binary, dilated_vertical_lines)

    # Optional: Apply a closing operation to fill in gaps
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned_binary = cv2.morphologyEx(cleaned_binary, cv2.MORPH_CLOSE, closing_kernel)

    # Apply median blur to remove small noise
    cleaned_binary = cv2.medianBlur(cleaned_binary, 3)

    # Repair fragmented text using dilation
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    repaired_binary = cv2.dilate(cleaned_binary, repair_kernel, iterations=1)
    repaired_binary = cv2.morphologyEx(repaired_binary, cv2.MORPH_CLOSE, repair_kernel)

    # Find contours on the cleaned binary image
    contours, _ = cv2.findContours(repaired_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create output directory
    output_dir = "/Users/ryan/Desktop/cropped_digits"
    os.makedirs(output_dir, exist_ok=True)

    def preprocess_to_mnist_format(image):
        """Convert an image to MNIST format (28x28, centered, grayscale)."""
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
        
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,2))
        dilated_canvas = cv2.dilate(canvas, vertical_kernel, iterations=1)
        return dilated_canvas

    # Extract bounding boxes and sort them by x-coordinate
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 5 < w < 50 and 17 < h < 60:
            bounding_boxes.append((x, y, w, h))

    bounding_boxes.sort(key=lambda box: box[0])

    # Process each bounding box
    for i, (x, y, w, h) in enumerate(bounding_boxes):
        digit_roi = cleaned_binary[y:y+h, x:x+w]
        mnist_digit = preprocess_to_mnist_format(digit_roi)
        digit_filename = os.path.join(output_dir, f"digit_{i+1}_mnist.png")
        cv2.imwrite(digit_filename, mnist_digit)
        cv2.rectangle(cleaned_binary, (x, y), (x + w, y + h), (255, 255, 255), 1) 

    # Display results
    cv2.imshow('Cleaned Image with Bounding Boxes', cleaned_binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Initialize the main tkinter window
root = tk.Tk()
root.title("Localize Small Boxes")

# Global variables
pdf_image = None
tk_image = None
scaled_image = None
image_scale = 1
start_x = start_y = 0
rect_id = None

# Create GUI elements
control_frame = tk.Frame(root)
control_frame.pack(pady=10)

upload_button = tk.Button(control_frame, text="Upload PDF", command=load_pdf)
upload_button.grid(row=0, column=0, padx=5)

canvas = tk.Canvas(root, width=600, height=800, bg="gray")
canvas.pack(pady=10)

# Bind mouse events to the canvas
canvas.bind("<ButtonPress-1>", start_crop)
canvas.bind("<B1-Motion>", update_crop)
canvas.bind("<ButtonRelease-1>", finish_crop)

# Maximize the window
root.state('zoomed')

# Start the main event loop
root.mainloop()

