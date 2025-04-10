from tkinter import filedialog, messagebox
from pdf2image import convert_from_path
import os

def load_pdf(pdf_files, pdf_combobox, debug_images, display_selected_pdf):
    """Load and display multiple one-page PDF files."""
    file_paths = filedialog.askopenfilenames(
        filetypes=[("PDF Files", "*.pdf")],
        title="Select one or more one-page PDF files"
    )
    if not file_paths:
        return
    pdf_files.clear()
    pdf_files.extend(file_paths)
    debug_images.clear()
    pdf_combobox['values'] = [os.path.basename(path) for path in pdf_files]
    if pdf_files:
        pdf_combobox.current(0)
        display_selected_pdf()

def display_selected_pdf(pdf_files, pdf_combobox, pdf_image, display_image, status_label):
    """Display the selected PDF from the combobox."""
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
        pdf_image[0] = images[0]  # Using a list to mimic global variable modification
        display_image(pdf_image[0])
        status_label.config(text=f"Loaded: {os.path.basename(pdf_files[selection])}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load PDF: {e}")