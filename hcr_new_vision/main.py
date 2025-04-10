import tkinter as tk
from tkinter import ttk
from pdf_utils import load_pdf, display_selected_pdf
from gui_utils import *
from classification import classify, export_to_excel

# Global variables as lists to mimic mutable globals across modules
pdf_image = [None]
tk_image = [None]
scaled_image = [None]
image_scale = [1]
start_x = [0]
start_y = [0]
rect_id = [None]
crop_coords = [None]
pdf_files = []
debug_images = {}
canvas_offset_x = [0]
canvas_offset_y = [0]

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

main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

control_frame = tk.Frame(main_frame)
control_frame.pack(fill=tk.X, pady=10)

upload_button = tk.Button(control_frame, text="Upload PDFs", command=lambda: load_pdf(pdf_files, pdf_combobox, debug_images, lambda: display_selected_pdf(pdf_files, pdf_combobox, pdf_image, lambda img: display_image(canvas, img, tk_image, scaled_image, image_scale, canvas_offset_x, canvas_offset_y), status_label)), width=15, height=2)
upload_button.grid(row=0, column=0, padx=5)

pdf_label = tk.Label(control_frame, text="Select PDF:")
pdf_label.grid(row=0, column=1, padx=5, pady=5, sticky="e")

pdf_combobox = ttk.Combobox(control_frame, width=40, state="readonly")
pdf_combobox.grid(row=0, column=2, padx=5, pady=5, sticky="w")
pdf_combobox.bind("<<ComboboxSelected>>", lambda e: on_combobox_select(e, lambda: display_selected_pdf(pdf_files, pdf_combobox, pdf_image, lambda img: display_image(canvas, img, tk_image, scaled_image, image_scale, canvas_offset_x, canvas_offset_y), status_label)))

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
    command=lambda: classify(root, pdf_image, crop_coords, pdf_files, rect_id, replace_var, user_entry_var, debug_images, status_label, canvas, lambda results: show_recognition_results(root, results, pdf_files, pdf_combobox, lambda fp: show_debug_window(root, fp, debug_images), export_to_excel)),
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

canvas.bind("<ButtonPress-1>", lambda e: start_crop(e, canvas, canvas_offset_x, start_x, start_y, rect_id))
canvas.bind("<B1-Motion>", lambda e: update_crop(e, canvas, rect_id, start_x, start_y, canvas_offset_x, canvas_offset_y))
canvas.bind("<ButtonRelease-1>", lambda e: finish_crop(e, canvas, pdf_image, scaled_image, image_scale, canvas_offset_x, canvas_offset_y, start_x, start_y, crop_coords, rect_id, status_label))
canvas.bind("<Configure>", lambda e: setup_canvas(pdf_image, lambda img: display_image(canvas, img, tk_image, scaled_image, image_scale, canvas_offset_x, canvas_offset_y)))

root.mainloop()