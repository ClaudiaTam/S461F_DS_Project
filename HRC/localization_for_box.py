import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk, ImageOps
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
    """Finish the crop selection and process the image."""
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

        # Save the cropped region temporarily
        temp_image_path = "temp_cropped_region.png"
        cropped_image.save(temp_image_path)

        # Perform localization
        localized_boxes = localization(temp_image_path)

        if localized_boxes:
            # Sort bounding boxes left to right
            localized_boxes.sort(key=lambda b: b[0])  # Sort by x-coordinate

            # Get extreme points
            leftmost_box = localized_boxes[0]
            rightmost_box = localized_boxes[-1]

            # Define extreme coordinates
            final_x1, final_y1 = leftmost_box[0], leftmost_box[1]  # Top-left
            final_x2, final_y2 = rightmost_box[0] + rightmost_box[2], rightmost_box[1] + rightmost_box[3]  # Bottom-right

            # Crop the large bounding region
            final_crop_box = (final_x1, final_y1, final_x2, final_y2)
            final_cropped_image = cropped_image.crop(final_crop_box)

            # Save the final cropped region
            final_image_path = "final_cropped_region.png"
            final_cropped_image.save(final_image_path)

            # Process the final cropped image to remove vertical lines
            process_image(final_image_path)

        else:
            messagebox.showinfo("No Regions Found", "No bounding regions were detected.")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to crop image: {e}")
    finally:
        # Delete the temporary file after use
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        # Delete the rectangle after finishing the crop
        if rect_id:
            canvas.delete(rect_id)

def localization(image_path, margin=2):
    """Localize handwritten text from the image and display bounding boxes."""
    # Load and preprocess the image
    image = cv2.imread(image_path)
    INV_image = cv2.bitwise_not(image)  # Invert the image
    gray = cv2.cvtColor(INV_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Thresholding to detect handwritten text
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Store bounding boxes
    bounding_boxes = []

    # Collect bounding boxes that meet size and aspect ratio criteria
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Filtering based on size and aspect ratio to exclude noise
        if 30 < w < 200 and 30 < h < 150:  # Adjust thresholds as needed
            bounding_boxes.append((x, y, w, h))

    # Filter out nested bounding boxes and boxes that touch the edges
    final_boxes = []
    image_height, image_width = image.shape[:2]

    for i, (x1, y1, w1, h1) in enumerate(bounding_boxes):
        is_nested = False
        for j, (x2, y2, w2, h2) in enumerate(bounding_boxes):
            if i != j:  # Don't compare a box with itself
                # Check if (x1, y1, w1, h1) is fully inside (x2, y2, w2, h2)
                if x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2:
                    is_nested = True
                    break
        # Check if the bounding box touches the edges of the cropped region
        if not is_nested and x1 >= margin and y1 >= margin and x1 + w1 <= image_width - margin and y1 + h1 <= image_height - margin:
            final_boxes.append((x1, y1, w1, h1))

    # Draw the final bounding boxes on the original image
    for (x, y, w, h) in final_boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box

    # Show the cropped image with bounding boxes for debugging
    cv2.imshow("Cropped Image with Bounding Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return final_boxes  # Return bounding boxes for any further processing if needed

def process_image(image_path):
    """Process the image to remove vertical lines and display the cleaned binary image."""
    # Load the image
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to get a binary image (invert colors)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Detect vertical lines using morphological operations
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 70))  # Adjusted kernel size
    eroded = cv2.erode(binary, vertical_kernel, iterations=1)
    detected_vertical_lines = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, vertical_kernel)

    # Subtract vertical lines from binary image to keep handwritten text
    dilated_vertical_lines = cv2.dilate(detected_vertical_lines, vertical_kernel, iterations=1)
    cleaned_binary = cv2.subtract(binary, dilated_vertical_lines)

    # Optional: Apply a closing operation to fill in gaps
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned_binary = cv2.morphologyEx(cleaned_binary, cv2.MORPH_CLOSE, closing_kernel)

    # Apply median blur to remove small noise
    cleaned_binary = cv2.medianBlur(cleaned_binary, 3)

    # Repair fragmented text using dilation
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Adjust kernel size as needed
    repaired_binary = cv2.dilate(cleaned_binary, repair_kernel, iterations=1)
    repaired_binary = cv2.morphologyEx(repaired_binary, cv2.MORPH_CLOSE, repair_kernel)

    # Find contours on the cleaned binary image
    contours, _ = cv2.findContours(repaired_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around detected words on the cleaned_binary image
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 2 < w < 50 and 15 < h < 60:  # Adjust thresholds as needed
            # Calculate the center of the bounding rectangle
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Determine the size of the square (use the maximum dimension)
            square_size = max(w, h)
            
            # Calculate the top-left corner of the square
            square_x = center_x - square_size // 2
            square_y = center_y - square_size // 2
            
            # Draw the square
            cv2.rectangle(cleaned_binary, (square_x, square_y), (square_x + square_size, square_y + square_size), (255, 255, 255), 2)  # Use white color for binary image

    # Display results
    cv2.imshow('Detected Vertical Lines', detected_vertical_lines)
    cv2.imshow('Cleaned Image with Bounding Boxes', cleaned_binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Initialize the main tkinter window
root = tk.Tk()
root.title("Localization")

# Global variables
pdf_image = None
tk_image = None
scaled_image = None
image_scale = 1
start_x = start_y = 0
rect_id = None

# Create GUI elements
frame = tk.Frame(root)
frame.pack(pady=10)

upload_button = tk.Button(frame, text="Upload PDF", command=load_pdf)
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
