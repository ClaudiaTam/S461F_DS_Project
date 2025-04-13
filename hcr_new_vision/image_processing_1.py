import cv2
import numpy as np
from PIL import Image


def preprocess_to_mnist_format(image):
    """Convert an image to MNIST format (28x28, centered, grayscale).

    Args:
        image: Input image (numpy array).

    Returns:
        np.ndarray: Processed 28x28 grayscale image.
    """
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(image)
    if coords is None:
        return np.zeros((28, 28), dtype=np.uint8)
    x, y, w, h = cv2.boundingRect(coords)
    if w == 0 or h == 0:
        return np.zeros((28, 28), dtype=np.uint8)
    digit = image[y:y + h, x:x + w]
    max_dim = max(w, h)
    scale = 20.0 / max_dim
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


def merge_nearby_contours(contours, distance_threshold=5):  # MODIFIED: Reduced threshold from 10 to 5
    """Merge nearby contours based on their bounding boxes.

    Args:
        contours: List of contours from cv2.findContours.
        distance_threshold: Maximum distance to merge contours (default: 5).

    Returns:
        list: Merged bounding boxes as (x, y, w, h).
    """
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    merged_boxes = []
    used = [False] * len(bounding_boxes)

    for i in range(len(bounding_boxes)):
        if used[i]:
            continue
        x1, y1, w1, h1 = bounding_boxes[i]
        merged_xmin = x1
        merged_ymin = y1
        merged_xmax = x1 + w1
        merged_ymax = y1 + h1
        used[i] = True

        for j in range(i + 1, len(bounding_boxes)):
            if used[j]:
                continue
            x2, y2, w2, h2 = bounding_boxes[j]
            dx = min(abs(x1 + w1 - x2), abs(x2 + w2 - x1)) if x1 < x2 + w2 and x2 < x1 + w1 else abs(x1 - x2)
            dy = min(abs(y1 + h1 - y2), abs(y2 + h2 - y1)) if y1 < y2 + h2 and y2 < y1 + h1 else abs(y1 - y2)
            distance = max(dx, dy)

            if distance < distance_threshold:
                merged_xmin = min(merged_xmin, x2)
                merged_ymin = min(merged_ymin, y2)
                merged_xmax = max(merged_xmax, x2 + w2)
                merged_ymax = max(merged_ymax, y2 + h2)
                used[j] = True

        merged_boxes.append((merged_xmin, merged_ymin, merged_xmax - merged_xmin, merged_ymax - merged_ymin))
    return merged_boxes


def process_image(image_path, output_dir):
    """Process an image to detect and crop characters for classification.

    Args:
        image_path: Path to the input image.
        output_dir: Directory to save cropped character images.

    Returns:
        tuple: (digit_results, cleaned_binary, mnist_images, green_boxes_img, binary_results)
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(
        enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 12
    )

    # Remove vertical and horizontal lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    eroded = cv2.erode(binary, vertical_kernel, iterations=1)
    detected_vertical_lines = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, vertical_kernel)
    dilated_vertical_lines = cv2.dilate(detected_vertical_lines, vertical_kernel, iterations=1)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    eroded_h = cv2.erode(binary, horizontal_kernel, iterations=1)
    detected_horizontal_lines = cv2.morphologyEx(eroded_h, cv2.MORPH_OPEN, horizontal_kernel)
    dilated_horizontal_lines = cv2.dilate(detected_horizontal_lines, horizontal_kernel, iterations=1)
    combined_lines = cv2.add(dilated_vertical_lines, dilated_horizontal_lines)
    cleaned_binary = cv2.subtract(binary, combined_lines)

    # Clean up noise
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned_binary = cv2.morphologyEx(cleaned_binary, cv2.MORPH_CLOSE, closing_kernel)
    cleaned_binary = cv2.medianBlur(cleaned_binary, 3)
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    repaired_binary = cv2.dilate(cleaned_binary, repair_kernel, iterations=1)

    # Additional dilation to connect disjointed parts
    connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    connected_binary = cv2.dilate(repaired_binary, connect_kernel, iterations=2)

    margin = 5
    contours, _ = cv2.findContours(connected_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_height, img_width = repaired_binary.shape

    initial_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # MODIFIED: Relaxed size constraints to capture smaller and larger characters
        if 2 < w < 150 and 10 < h < 150:  # Changed w: 100 -> 150, h: 17 -> 10 and 100 -> 150
            if (x > margin and y > margin and
                    x + w < img_width - margin and
                    y + h < img_height - margin):
                initial_boxes.append(contour)

    bounding_boxes = merge_nearby_contours(initial_boxes, distance_threshold=5)  # Updated threshold
    bounding_boxes.sort(key=lambda box: box[0])

    # MODIFIED: Filter out large gaps (spaces) between bounding boxes
    filtered_boxes = []
    for i in range(len(bounding_boxes)):
        if i == 0:
            filtered_boxes.append(bounding_boxes[i])
            continue
        prev_box = bounding_boxes[i - 1]
        curr_box = bounding_boxes[i]
        gap = curr_box[0] - (prev_box[0] + prev_box[2])  # Calculate gap between boxes
        if gap < 20:  # If gap is less than 20 pixels, keep the box (adjust threshold as needed)
            filtered_boxes.append(curr_box)
    bounding_boxes = filtered_boxes

    digit_results = []
    binary_results = []
    mnist_images = []
    green_boxes_img = cv2.cvtColor(cleaned_binary.copy(), cv2.COLOR_GRAY2BGR)

    for i, (x, y, w, h) in enumerate(bounding_boxes):
        digit_roi = cleaned_binary[y:y + h, x:x + w]
        mnist_digit = preprocess_to_mnist_format(digit_roi)
        mnist_images.append(mnist_digit)
        digit_filename = os.path.join(output_dir, f"char_{i + 1}_mnist.png")
        cv2.imwrite(digit_filename, mnist_digit)
        # Note: Classification is handled in classification.py
        cv2.rectangle(green_boxes_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return digit_results, cleaned_binary, mnist_images, green_boxes_img, binary_results