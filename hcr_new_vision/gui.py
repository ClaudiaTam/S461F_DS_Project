import os
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from PIL import Image, ImageTk
from pdf2image import convert_from_path

from classification import classify_image
from image_processing import process_image
from utils import export_to_excel, download_image, download_mnist_images

class HandwrittenRecognitionApp:
    """Main application class for handwritten character recognition GUI."""
    def __init__(self, root, digit_model, letter_model, binary_model):
        self.root = root
        self.digit_model = digit_model
        self.letter_model = letter_model
        self.binary_model = binary_model
        self.root.title("HRC - Handwritten Recognition")
        self.root.state('zoomed')

        # Initialize GUI variables
        self.pdf_image = None
        self.tk_image = None
        self.scaled_image = None
        self.image_scale = 1
        self.rect_id = None
        self.crop_coords = None
        self.pdf_files = []
        self.debug_images = {}
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0
        self.zoom_factor = 1.0
        self.original_image = None
        self.zoom_mode = False
        self.pan_start_x = None
        self.pan_start_y = None

        self.setup_gui()

    def setup_gui(self):
        """Set up the main GUI components."""
        # Configure style
        try:
            style = ttk.Style()
            available_themes = style.theme_names()
            if 'clam' in available_themes:
                style.theme_use('clam')
            elif 'vista' in available_themes:
                style.theme_use('vista')
        except Exception:
            pass

        # Main frame
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Control frame
        self.control_frame = tk.Frame(self.main_frame)
        self.control_frame.pack(fill=tk.X, pady=10)

        # Upload button
        tk.Button(
            self.control_frame, text="Upload PDFs", command=self.load_pdf,
            width=15, height=2
        ).grid(row=0, column=0, padx=5)

        # PDF selection
        tk.Label(self.control_frame, text="Select PDF:").grid(
            row=0, column=1, padx=5, pady=5, sticky="e"
        )
        self.pdf_combobox = ttk.Combobox(self.control_frame, width=40, state="readonly")
        self.pdf_combobox.grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.pdf_combobox.bind("<<ComboboxSelected>>", self.on_combobox_select)

        # Classify button
        tk.Button(
            self.control_frame, text="Classify", command=self.classify,
            width=15, height=2, bg="#4CAF50", fg="black", font=("Arial", 10, "bold")
        ).grid(row=0, column=3, padx=5, pady=5)

        # Status label
        self.status_label = tk.Label(
            self.control_frame, text="Ready. Please load a PDF file.",
            font=("Arial", 10), bd=1, relief=tk.SUNKEN, anchor=tk.W
        )
        self.status_label.grid(row=0, column=4, columnspan=1, padx=5, pady=5, sticky="ew")

        # Zoom controls
        self.zoom_button = tk.Button(
            self.control_frame, text="Switch to Zoom Mode", command=self.toggle_zoom_mode,
            width=15, height=2, bg="#2196F3", fg="black", font=("Arial", 10, "bold")
        )
        self.zoom_button.grid(row=0, column=5, padx=5, pady=5)

        tk.Button(
            self.control_frame, text="Reset Zoom", command=self.reset_zoom,
            width=10, height=2, bg="#9E9E9E", fg="black", font=("Arial", 10, "bold")
        ).grid(row=0, column=6, padx=5, pady=5)

        # Canvas frame
        self.canvas_frame = tk.Frame(self.main_frame, bd=2, relief=tk.SUNKEN)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.canvas = tk.Canvas(self.canvas_frame, bg="light gray")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbars
        self.h_scrollbar = ttk.Scrollbar(self.canvas_frame, orient="horizontal", command=self.canvas.xview)
        self.v_scrollbar = ttk.Scrollbar(self.canvas_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.h_scrollbar.set, yscrollcommand=self.v_scrollbar.set)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Instructions
        tk.Label(
            self.main_frame,
            text="Instructions: Upload a PDF, select an area with digits by drawing a rectangle, then click 'Classify'",
            font=("Arial", 10), fg="#666666"
        ).pack(pady=5, anchor=tk.W)

        # Bind canvas events
        self.canvas.bind("<ButtonPress-1>", self.start_crop)
        self.canvas.bind("<B1-Motion>", self.update_crop)
        self.canvas.bind("<ButtonRelease-1>", self.finish_crop)
        self.canvas.bind("<Configure>", lambda e: self.setup_canvas())

    def load_pdf(self):
        """Load and display multiple one-page PDF files."""
        file_paths = filedialog.askopenfilenames(
            filetypes=[("PDF Files", "*.pdf")], title="Select one or more one-page PDF files"
        )
        if not file_paths:
            return
        self.pdf_files = list(file_paths)
        self.debug_images.clear()
        self.pdf_combobox['values'] = [os.path.basename(path) for path in self.pdf_files]
        if self.pdf_files:
            self.pdf_combobox.current(0)
            self.display_selected_pdf()

    def display_selected_pdf(self):
        """Display the selected PDF from the combobox."""
        if not self.pdf_files:
            return
        selection = self.pdf_combobox.current()
        if selection < 0:
            return
        try:
            images = convert_from_path(self.pdf_files[selection], dpi=200)
            if len(images) != 1:
                messagebox.showerror("Error", "Please upload one-page PDF files.")
                return
            self.pdf_image = images[0]
            self.display_image(self.pdf_image)
            self.status_label.config(text=f"Loaded: {os.path.basename(self.pdf_files[selection])}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load PDF: {e}")

    def display_image(self, image):
        """Display the given PIL image in the GUI with zoom support."""
        self.canvas.delete("all")
        self.original_image = image
        image_width, image_height = image.size
        canvas_width, canvas_height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if canvas_width <= 1:
            canvas_width, canvas_height = 600, 800

        scale_factor = min(canvas_width / image_width, canvas_height / image_height) * self.zoom_factor
        self.image_scale = scale_factor
        new_width = int(image_width * scale_factor)
        new_height = int(image_height * scale_factor)
        self.scaled_image = image.resize((new_width, new_height))

        self.canvas_offset_x = max(0, (canvas_width - new_width) // 2)
        self.canvas_offset_y = max(0, (canvas_height - new_height) // 2)

        self.tk_image = ImageTk.PhotoImage(self.scaled_image)
        self.canvas.create_image(self.canvas_offset_x, self.canvas_offset_y, anchor="nw", image=self.tk_image)

        if self.zoom_factor > 1.0:
            self.canvas.configure(scrollregion=(0, 0, max(canvas_width, new_width), max(canvas_height, new_height)))
        else:
            self.canvas.configure(scrollregion=(0, 0, canvas_width, canvas_height))

    def on_mousewheel(self, event):
        """Handle mouse wheel events for zooming."""
        if not self.zoom_mode or not self.original_image:
            return
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        if event.delta > 0 or (hasattr(event, 'num') and event.num == 4):
            self.zoom_factor = min(self.zoom_factor * 1.1, 1.1)
        else:
            self.zoom_factor = max(self.zoom_factor * 0.9, 0.5)
        self.display_image(self.original_image)
        self.status_label.config(text=f"Zoom: {self.zoom_factor:.1f}x")

    def start_pan(self, event):
        """Start panning the canvas when zoom level > 1."""
        if self.zoom_factor > 1.0:
            self.canvas.config(cursor="fleur")
            self.pan_start_x = event.x
            self.pan_start_y = event.y

    def update_pan(self, event):
        """Pan the canvas during mouse drag when zoomed in."""
        if self.zoom_factor > 1.0 and self.pan_start_x is not None:
            dx = self.pan_start_x - event.x
            dy = self.pan_start_y - event.y
            self.canvas.xview_scroll(dx, "units")
            self.canvas.yview_scroll(dy, "units")
            self.pan_start_x = event.x
            self.pan_start_y = event.y

    def end_pan(self, event):
        """End panning and reset cursor."""
        self.canvas.config(cursor="arrow")
        self.pan_start_x = None
        self.pan_start_y = None

    def toggle_zoom_mode(self):
        """Toggle between crop and zoom mode."""
        self.zoom_mode = not self.zoom_mode
        if self.zoom_mode:
            self.canvas.unbind("<ButtonPress-1>")
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")
            self.canvas.bind("<ButtonPress-1>", self.start_pan)
            self.canvas.bind("<B1-Motion>", self.update_pan)
            self.canvas.bind("<ButtonRelease-1>", self.end_pan)
            self.canvas.bind("<MouseWheel>", self.on_mousewheel)
            self.canvas.bind("<Button-4>", self.on_mousewheel)
            self.canvas.bind("<Button-5>", self.on_mousewheel)
            self.zoom_button.config(text="Switch to Crop Mode", bg="#FF9800")
            self.status_label.config(text="Zoom Mode: Use trackpad/wheel to zoom, click and drag to pan")
        else:
            self.canvas.unbind("<ButtonPress-1>")
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")
            self.canvas.unbind("<MouseWheel>")
            self.canvas.unbind("<Button-4>")
            self.canvas.unbind("<Button-5>")
            self.canvas.bind("<ButtonPress-1>", self.start_crop)
            self.canvas.bind("<B1-Motion>", self.update_crop)
            self.canvas.bind("<ButtonRelease-1>", self.finish_crop)
            self.zoom_button.config(text="Switch to Zoom Mode", bg="#2196F3")
            self.status_label.config(text="Crop Mode: Click and drag to select an area")

    def reset_zoom(self):
        """Reset zoom to 100%."""
        self.zoom_factor = 1.0
        if self.original_image:
            self.display_image(self.original_image)
        self.status_label.config(text="Zoom reset to 100%")

    def start_crop(self, event):
        """Start selecting rectangle area for cropping."""
        if self.rect_id is not None:
            self.canvas.delete(self.rect_id)
            self.rect_id = None
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        self.start_x = canvas_x - self.canvas_offset_x
        self.start_y = canvas_y - self.canvas_offset_y
        self.rect_id = self.canvas.create_rectangle(
            canvas_x, canvas_y, canvas_x, canvas_y, outline="red", width=2
        )

    def update_crop(self, event):
        """Update the cropping rectangle while dragging."""
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        self.canvas.coords(
            self.rect_id, self.start_x + self.canvas_offset_x, self.start_y + self.canvas_offset_y,
            canvas_x, canvas_y
        )

    def finish_crop(self, event):
        """Finish cropping and store coordinates."""
        if self.pdf_image is None:
            messagebox.showerror("Error", "No PDF image loaded.")
            return
        try:
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)
            end_x = canvas_x - self.canvas_offset_x
            end_y = canvas_y - self.canvas_offset_y
            x1, y1, x2, y2 = min(self.start_x, end_x), min(self.start_y, end_y), max(self.start_x, end_x), max(self.start_y, end_y)
            x1 = max(0, min(x1, self.scaled_image.width))
            y1 = max(0, min(y1, self.scaled_image.height))
            x2 = max(0, min(x2, self.scaled_image.width))
            y2 = max(0, min(y2, self.scaled_image.height))
            x1_orig = int(x1 / self.image_scale)
            y1_orig = int(y1 / self.image_scale)
            x2_orig = int(x2 / self.image_scale)
            y2_orig = int(y2 / self.image_scale)
            self.crop_coords = (x1_orig, y1_orig, x2_orig, y2_orig)
            self.status_label.config(text=f"Selection complete. Region: {self.crop_coords}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to calculate crop coordinates: {e}")

    def classify(self):
        """Classify the cropped region of the PDF image and store debug images."""
        if self.pdf_image is None:
            messagebox.showerror("Error", "No PDF image loaded.")
            return
        if self.crop_coords is None:
            messagebox.showerror("Error", "Please draw a bounding box first.")
            return

        self.status_label.config(text="Processing... Please wait.")
        self.root.update()

        if self.rect_id is not None:
            self.canvas.delete(self.rect_id)
            self.rect_id = None

        recognition_results = []
        output_dir = "/Users/ryan/Desktop/cropped_digits"
        os.makedirs(output_dir, exist_ok=True)

        for file_path in self.pdf_files:
            try:
                images = convert_from_path(file_path, dpi=200)
                if len(images) != 1:
                    recognition_results.append(f"{os.path.basename(file_path)}: Not a one-page PDF.")
                    continue
                original_img = images[0]
                cropped_image = original_img.crop(self.crop_coords)
                temp_image_path = "temp_cropped_region.png"
                cropped_image.save(temp_image_path)

                debug_entry = {
                    'original': original_img,
                    'cropped': cropped_image,
                    'processed': None,
                    'mnist_images': None,
                    'green_boxes_img': None,
                    'binary_results': None
                }

                digit_results, processed_img, mnist_images, green_boxes_img, binary_results = process_image(temp_image_path, output_dir)
                processed_img_pil = Image.fromarray(processed_img)
                green_boxes_img_pil = Image.fromarray(green_boxes_img)

                # Classify each MNIST-formatted image
                for i, mnist_img in enumerate(mnist_images):
                    digit_filename = os.path.join(output_dir, f"char_{i+1}_mnist.png")
                    result, binary_label = classify_image(
                        digit_filename, self.digit_model, self.letter_model, self.binary_model
                    )
                    if result is not None:
                        digit_results.append(result)
                        binary_results.append(binary_label)

                debug_entry.update({
                    'processed': processed_img_pil,
                    'mnist_images': mnist_images,
                    'results': ''.join(digit_results),
                    'green_boxes_img': green_boxes_img_pil,
                    'binary_results': binary_results
                })

                recognition_results.append(f"{os.path.basename(file_path)}: {''.join(digit_results)}")
                self.debug_images[file_path] = debug_entry

                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
            except Exception as e:
                recognition_results.append(f"{os.path.basename(file_path)}: Error processing file - {e}")

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        self.status_label.config(text="Classification complete!")
        self.show_recognition_results(recognition_results)

    def show_recognition_results(self, results):
        """Display recognition results in a scrollable text window."""
        result_window = tk.Toplevel(self.root)
        result_window.title("Recognition Results")
        result_window.geometry("1200x400")

        control_frame = tk.Frame(result_window)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(control_frame, text="Select PDF for Debug:").pack(side=tk.LEFT, padx=5)
        pdf_debug_combobox = ttk.Combobox(control_frame, width=40, state="readonly")
        pdf_debug_combobox.pack(side=tk.LEFT, padx=5)
        pdf_debug_combobox['values'] = [os.path.basename(path) for path in self.pdf_files]
        if self.pdf_files:
            pdf_debug_combobox.current(self.pdf_combobox.current())

        def on_debug_select():
            selected_index = pdf_debug_combobox.current()
            if selected_index >= 0:
                self.show_debug_window(self.pdf_files[selected_index])

        tk.Button(control_frame, text="Show Debug Details", command=on_debug_select).pack(side=tk.LEFT, padx=5)
        tk.Button(
            control_frame, text="Output to Excel",
            command=lambda: export_to_excel(results, result_window)
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Close", command=result_window.destroy).pack(side=tk.RIGHT, padx=5)

        text_area = scrolledtext.ScrolledText(result_window, wrap=tk.WORD, width=60, height=20, font=("Arial", 14))
        text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        for result in results:
            text_area.insert(tk.END, result + "\n")
        text_area.configure(state="disabled")

    def show_debug_window(self, file_path):
        """Show debug window with processing details."""
        if not file_path or file_path not in self.debug_images:
            messagebox.showinfo("Debug Info", "No debug information available.")
            return

        debug_window = tk.Toplevel(self.root)
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

        debug_data = self.debug_images[file_path]
        row = 0

        if 'results' in debug_data:
            tk.Label(
                scrollable_frame, text=f"Recognition Result: {debug_data['results']}",
                font=("Arial", 14, "bold")
            ).grid(row=row, column=0, columnspan=2, pady=10, sticky="w", padx=10)
            row += 1

        # Cropped Image
        if 'cropped' in debug_data and debug_data['cropped']:
            cropped_frame = tk.Frame(scrollable_frame)
            cropped_frame.grid(row=row, column=0, sticky="w", padx=10, pady=5)
            tk.Label(cropped_frame, text="Cropped Image:", font=("Arial", 12)).pack(side=tk.LEFT, padx=5)
            tk.Button(
                cropped_frame, text="Download",
                command=lambda: download_image(debug_data['cropped'], "cropped", debug_window, file_path),
                width=10, height=1, bg="#ffffff", fg="black", font=("Arial", 10, "bold")
            ).pack(side=tk.LEFT, padx=5)
            row += 1
            img = debug_data['cropped'].copy()
            img.thumbnail((400, 400))
            tk_img = ImageTk.PhotoImage(img)
            label = tk.Label(scrollable_frame, image=tk_img)
            label.image = tk_img
            label.grid(row=row, column=0, padx=10, pady=5, sticky W")
            row += 1

        # Processed Image
        if 'processed' in debug_data and debug_data['processed']:
            processed_frame = tk.Frame(scrollable_frame)
            processed_frame.grid(row=row, column=0, sticky="w", padx=10, pady=5)
            tk.Label(processed_frame, text="Processed Image (After Preprocessing):", font=("Arial", 12)).pack(side=tk.LEFT, padx=5)
            tk.Button(
                processed_frame, text="Download",
                command=lambda: download_image(debug_data['processed'], "processed", debug_window, file_path),
                width=10, height=1, bg="#ffffff", fg="black", font=("Arial", 10, "bold")
            ).pack(side=tk.LEFT, padx=5)
            row += 1
            img = debug_data['processed'].copy()
            img.thumbnail((400, 400))
            tk_img = ImageTk.PhotoImage(img)
            label = tk.Label(scrollable_frame, image=tk_img)
            label.image = tk_img
            label.grid(row=row, column=0, padx=10, pady=5, sticky="w")
            row += 1

        # Digit Detection
        if 'green_boxes_img' in debug_data and debug_data['green_boxes_img']:
            detection_frame = tk.Frame(scrollable_frame)
            detection_frame.grid(row=row, column=0, sticky="w", padx=10, pady=5)
            tk.Label(
                detection_frame, text="Handwritten Characters Detection (Green Bounding Boxes):",
                font=("Arial", 12)
            ).pack(side=tk.LEFT, padx=5)
            tk.Button(
                detection_frame, text="Download",
                command=lambda: download_image(debug_data['green_boxes_img'], "digit_detection", debug_window, file_path),
                width=10, height=1, bg="#ffffff", fg="black", font=("Arial", 10, "bold")
            ).pack(side=tk.LEFT, padx=5)
            row += 1
            img = debug_data['green_boxes_img'].copy()
            img.thumbnail((400, 400))
            tk_img = ImageTk.PhotoImage(img)
            label = tk.Label(scrollable_frame, image=tk_img)
            label.image = tk_img
            label.grid(row=row, column=0, padx=10, pady=5, sticky="w")
            row += 1

        # MNIST Images
        if 'mnist_images' in debug_data and debug_data['mnist_images']:
            mnist_title_frame = tk.Frame(scrollable_frame)
            mnist_title_frame.grid(row=row, column=0, sticky="w", padx=10, pady=5)
            tk.Label(
                mnist_title_frame, text="MNIST-Formatted Images (Used for Classification):",
                font=("Arial", 12)
            ).pack(side=tk.LEFT, padx=5)
            tk.Button(
                mnist_title_frame, text="Download",
                command=lambda: download_mnist_images(debug_data, debug_window, file_path),
                width=10, height=1, bg="#ffffff", fg="black", font=("Arial", 10, "bold")
            ).pack(side=tk.LEFT, padx=5)
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
                        digit_frame, text=f"Type: {debug_data['binary_results'][i]}",
                        font=("Arial", 10, "bold"), fg="blue"
                    ).pack()
                if 'results' in debug_data and i < len(debug_data['results']):
                    tk.Label(
                        digit_frame, text=f"Result: {debug_data['results'][i]}",
                        font=("Arial", 10, "bold")
                    ).pack()
            row += 1

        tk.Button(scrollable_frame, text="Close", command=debug_window.destroy, width=20, height=2).grid(
            row=row, column=0, pady=20
        )

    def setup_canvas(self):
        """Configure the canvas when the window is resized."""
        if self.pdf_image:
            self.display_image(self.pdf_image)

    def on_combobox_select(self, event):
        """Handle selection change in the PDF combobox."""
        self.display_selected_pdf()