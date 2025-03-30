import cv2
import numpy as np
from PIL import Image
import os

def preprocess_image(image_path):
    """Preprocess the image and detect characters."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(
        enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 12
    )

    # Remove lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    eroded_v = cv2.erode(binary, vertical_kernel, iterations=1)
    eroded_h = cv2.erode(binary, horizontal_kernel, iterations=1)
    lines_v = cv2.dilate(cv2.morphologyEx(eroded_v, cv2.MORPH_OPEN, vertical_kernel), vertical_kernel)
    lines_h = cv2.dilate(cv2.morphologyEx(eroded_h, cv2.MORPH_OPEN, horizontal_kernel), horizontal_kernel)
    cleaned_binary = cv2.subtract(binary, cv2.add(lines_v, lines_h))

    # Clean noise and connect characters
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned_binary = cv2.morphologyEx(cleaned_binary, cv2.MORPH_CLOSE, closing_kernel)
    cleaned_binary = cv2.medianBlur(cleaned_binary, 3)
    connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    connected_binary = cv2.dilate(cleaned_binary, connect_kernel, iterations=2)

    return connected_binary, cleaned_binary

def to_mnist_format(image):
    """Convert an image to MNIST format (28x28, centered, grayscale)."""
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
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset, y_offset = (28 - new_w) // 2, (28 - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

def detect_characters(binary_img, cleaned_binary, output_dir="C:/Users/guest0701", is_course_code=False):
    """Detect and crop characters from the binary image.

    Args:
        binary_img: Binary image after preprocessing.
        cleaned_binary: Cleaned binary image for visualization.
        output_dir: Directory to save intermediate images.
        is_course_code: If True, use tighter segmentation for course codes.
    """
    os.makedirs(output_dir, exist_ok=True)
    margin = 5
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_height, img_width = binary_img.shape

    def merge_nearby_contours(contours, distance_threshold=5 if is_course_code else 10):
        """Merge nearby contours with an adjustable distance threshold."""
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        merged_boxes, used = [], [False] * len(bounding_boxes)
        for i, (x1, y1, w1, h1) in enumerate(bounding_boxes):
            if used[i]:
                continue
            merged_xmin, merged_ymin = x1, y1
            merged_xmax, merged_ymax = x1 + w1, y1 + h1
            used[i] = True
            for j, (x2, y2, w2, h2) in enumerate(bounding_boxes[i+1:], i+1):
                if used[j]:
                    continue
                dx = min(abs(x1 + w1 - x2), abs(x2 + w2 - x1)) if x1 < x2 + w2 and x2 < x1 + w1 else abs(x1 - x2)
                dy = min(abs(y1 + h1 - y2), abs(y2 + h2 - y1)) if y1 < y2 + h2 and y2 < y1 + h1 else abs(y1 - y2)
                if max(dx, dy) < distance_threshold:
                    merged_xmin, merged_ymin = min(merged_xmin, x2), min(merged_ymin, y2)
                    merged_xmax, merged_ymax = max(merged_xmax, x2 + w2), max(merged_ymax, y2 + h2)
                    used[j] = True
            merged_boxes.append((merged_xmin, merged_ymin, merged_xmax - merged_xmin, merged_ymax - merged_ymin))
        return merged_boxes

    initial_boxes = [
        cv2.boundingRect(c) for c in contours
        if 2 < cv2.boundingRect(c)[2] < 100 and 17 < cv2.boundingRect(c)[3] < 100 and
           cv2.boundingRect(c)[0] > margin and cv2.boundingRect(c)[1] > margin and
           cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] < img_width - margin and
           cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] < img_height - margin
    ]
    bounding_boxes = merge_nearby_contours(initial_boxes)
    bounding_boxes.sort(key=lambda box: box[0])

    mnist_images = []
    green_boxes_img = cv2.cvtColor(cleaned_binary, cv2.COLOR_GRAY2BGR)

    for i, (x, y, w, h) in enumerate(bounding_boxes):
        char_roi = cleaned_binary[y:y+h, x:x+w]
        mnist_char = to_mnist_format(char_roi)
        mnist_images.append(mnist_char)
        char_filename = os.path.join(output_dir, f"char_{i+1}_mnist.png")
        cv2.imwrite(char_filename, mnist_char)
        cv2.rectangle(green_boxes_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return bounding_boxes, mnist_images, green_boxes_img