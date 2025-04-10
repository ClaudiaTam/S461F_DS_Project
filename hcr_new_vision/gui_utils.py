import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
import time
import os

def display_image(canvas, image, tk_image, scaled_image, image_scale, canvas_offset_x, canvas_offset_y):
    """Display the given PIL image in the GUI."""
    canvas.delete("all")
    image_width, image_height = image.size
    canvas_width, canvas_height = canvas.winfo_width(), canvas.winfo_height()
    if canvas_width <= 1:  # Canvas not yet drawn
        canvas_width, canvas_height = 600, 800
    scale_factor = min(canvas_width / image_width, canvas_height / image_height)
    new_width = int(image_width * scale_factor)
    new_height = int(image_height * scale_factor)
    scaled_image[0] = image.resize((new_width, new_height))
    image_scale[0] = scale_factor
    canvas_offset_x[0] = (canvas_width - new_width) // 2
    canvas_offset_y[0] = (canvas_height - new_height) // 2
    tk_image[0] = ImageTk.PhotoImage(scaled_image[0])
    canvas.create_image(canvas_offset_x[0], canvas_offset_y[0], anchor="nw", image=tk_image[0])

def start_crop(event, canvas, canvas_offset_x, start_x, start_y, rect_id):
    """Start cropping rectangle."""
    if rect_id[0] is not None:
        canvas.delete(rect_id[0])
        rect_id[0] = None
    start_x[0], start_y[0] = event.x - canvas_offset_x[0], event.y - canvas_offset_x[0]
    rect_id[0] = canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="red", width=2)

def update_crop(event, canvas, rect_id, start_x, start_y, canvas_offset_x, canvas_offset_y):
    """Update cropping rectangle."""
    end_x = max(min(event.x, canvas.winfo_width()), 0)
    end_y = max(min(event.y, canvas.winfo_height()), 0)
    canvas.coords(rect_id[0], start_x[0] + canvas_offset_x[0], start_y[0] + canvas_offset_y[0], end_x, end_y)

def finish_crop(event, canvas, pdf_image, scaled_image, image_scale, canvas_offset_x, canvas_offset_y, start_x, start_y, crop_coords, rect_id, status_label):
    """Finish cropping and calculate coordinates."""
    if pdf_image[0] is None:
        messagebox.showerror("Error", "No PDF image loaded.")
        return

    try:
        end_x, end_y = event.x - canvas_offset_x[0], event.y - canvas_offset_y[0]
        x1, y1, x2, y2 = min(start_x[0], end_x), min(start_y[0], end_y), max(start_x[0], end_x), max(start_y[0], end_y)
        x1 = max(0, min(x1, scaled_image[0].width))
        y1 = max(0, min(y1, scaled_image[0].height))
        x2 = max(0, min(x2, scaled_image[0].width))
        y2 = max(0, min(y2, scaled_image[0].height))
        x1_orig = int(x1 / image_scale[0])
        y1_orig = int(y1 / image_scale[0])
        x2_orig = int(x2 / image_scale[0])
        y2_orig = int(y2 / image_scale[0])
        crop_coords[0] = (x1_orig, y1_orig, x2_orig, y2_orig)
        status_label.config(text="Selection complete. Ready to classify.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to calculate crop coordinates: {e}")

def show_recognition_results(root, results, pdf_files, pdf_combobox, show_debug_window, export_to_excel):
    """Display recognition results in a scrollable text window with an output button."""
    result_window = tk.Toplevel(root)
    result_window.title("Recognition Results")
    result_window.geometry("1200x400")
    
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

    output_btn = tk.Button(
        control_frame, 
        text="Output to Excel", 
        command=lambda: export_to_excel(results, result_window)
    )
    output_btn.pack(side=tk.LEFT, padx=5)
    
    close_btn = tk.Button(control_frame, text="Close", command=result_window.destroy)
    close_btn.pack(side=tk.RIGHT, padx=5)
    
    text_area = scrolledtext.ScrolledText(result_window, wrap=tk.WORD, width=60, height=20, font=("Arial", 14))
    text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    for result in results:
        text_area.insert(tk.END, result + "\n")
    text_area.configure(state="disabled")

def download_image(image, filename_prefix, parent_window, file_path=None):
    """Prompt user to save a single image with a unique filename."""
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
        messagebox.showinfo("Download Complete", f"{filename_prefix} image saved successfully to:\n{save_dir}")
    except Exception as e:
        messagebox.showerror("Download Error", f"Failed to save {filename_prefix} image: {e}")

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

def show_debug_window(root, file_path, debug_images):
    """Show a debug window with processing details including binary classification and additional download buttons."""
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
        cropped_frame = tk.Frame(scrollable_frame)
        cropped_frame.grid(row=row, column=0, sticky="w", padx=10, pady=5)
        
        tk.Label(
            cropped_frame, 
            text="Cropped Image:", 
            font=("Arial", 12)
        ).pack(side=tk.LEFT, padx=5)
        
        download_cropped_btn = tk.Button(
            cropped_frame, 
            text="Download", 
            command=lambda: download_image(debug_data['cropped'], "cropped", debug_window, file_path),
            width=10,
            height=1,
            bg="#ffffff",
            fg="black",
            font=("Arial", 10, "bold")
        )
        download_cropped_btn.pack(side=tk.LEFT, padx=5)
        
        row += 1
        
        img = debug_data['cropped'].copy()
        img.thumbnail((400, 400))
        tk_img = ImageTk.PhotoImage(img)
        label = tk.Label(scrollable_frame, image=tk_img)
        label.image = tk_img
        label.grid(row=row, column=0, padx=10, pady=5, sticky="w")
        row += 1
    
    if 'processed' in debug_data and debug_data['processed']:
        processed_frame = tk.Frame(scrollable_frame)
        processed_frame.grid(row=row, column=0, sticky="w", padx=10, pady=5)
        
        tk.Label(
            processed_frame, 
            text="Processed Image (After Preprocessing):", 
            font=("Arial", 12)
        ).pack(side=tk.LEFT, padx=5)
        
        download_processed_btn = tk.Button(
            processed_frame, 
            text="Download", 
            command=lambda: download_image(debug_data['processed'], "processed", debug_window, file_path),
            width=10,
            height=1,
            bg="#ffffff",
            fg="black",
            font=("Arial", 10, "bold")
        )
        download_processed_btn.pack(side=tk.LEFT, padx=5)
        
        row += 1
        
        img = debug_data['processed'].copy()
        img.thumbnail((400, 400))
        tk_img = ImageTk.PhotoImage(img)
        label = tk.Label(scrollable_frame, image=tk_img)
        label.image = tk_img
        label.grid(row=row, column=0, padx=10, pady=5, sticky="w")
        row += 1
    
    if 'green_boxes_img' in debug_data and debug_data['green_boxes_img']:
        detection_frame = tk.Frame(scrollable_frame)
        detection_frame.grid(row=row, column=0, sticky="w", padx=10, pady=5)
        
        tk.Label(
            detection_frame, 
            text="Digit Detection (Green Bounding Boxes):", 
            font=("Arial", 12)
        ).pack(side=tk.LEFT, padx=5)
        
        download_detection_btn = tk.Button(
            detection_frame, 
            text="Download", 
            command=lambda: download_image(debug_data['green_boxes_img'], "digit_detection", debug_window, file_path),
            width=10,
            height=1,
            bg="#ffffff",
            fg="black",
            font=("Arial", 10, "bold")
        )
        download_detection_btn.pack(side=tk.LEFT, padx=5)
        
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
            
            if 'binary_results' in debug_data and i < len(debug_data['binary_results']):
                tk.Label(
                    digit_frame, 
                    text=f"Type: {debug_data['binary_results'][i]}", 
                    font=("Arial", 10, "bold"),
                    fg="blue"
                ).pack()
            if 'results' in debug_data and i < len(debug_data['results']):
                tk.Label(
                    digit_frame, 
                    text=f"Result: {debug_data['results'][i]}", 
                    font=("Arial", 10, "bold")
                ).pack()
        
        row += 1

    tk.Button(scrollable_frame, text="Close", command=debug_window.destroy, width=20, height=2).grid(row=row, column=0, pady=20)

def setup_canvas(pdf_image, display_image):
    """Configure the canvas when the window is resized."""
    if pdf_image[0]:
        display_image(pdf_image[0])

def on_combobox_select(event, display_selected_pdf):
    """Handle selection change in the PDF combobox."""
    display_selected_pdf()