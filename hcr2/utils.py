from pdf2image import convert_from_path
import openpyxl
import time
import os
from tkinter import messagebox, filedialog

def pdf_to_image(pdf_path):
    """Convert a PDF to a PIL image."""
    try:
        images = convert_from_path(pdf_path, dpi=200)
        if len(images) != 1:
            raise ValueError("Please upload one-page PDF files.")
        return images[0]
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load PDF: {e}")
        return None