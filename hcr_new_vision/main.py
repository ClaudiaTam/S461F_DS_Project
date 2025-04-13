import tkinter as tk
from gui import HandwrittenRecognitionApp
from classification import load_models

def main():
    """Main entry point for the Handwritten Recognition application."""
    # Load models
    digit_model, letter_model, binary_model = load_models()

    # Initialize GUI
    root = tk.Tk()
    app = HandwrittenRecognitionApp(root, digit_model, letter_model, binary_model)
    root.mainloop()

if __name__ == "__main__":
    main()