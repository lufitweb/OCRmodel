import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os

def adjust_box(x1, y1, x2, y2, scale_factor=1.0):
    """
    Adjust the box coordinates based on a scale factor.
    scale_factor > 1 enlarges the box, < 1 shrinks it.
    """
    width = x2 - x1
    height = y2 - y1
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    new_x1 = max(0, center_x - new_width // 2)
    new_y1 = max(0, center_y - new_height // 2)
    new_x2 = new_x1 + new_width
    new_y2 = new_y1 + new_height
    return new_x1, new_y1, new_x2, new_y2

# Set up folders
input_folder = 'test_images'
output_folder = 'output'
output_whiteout_folder = 'output_whiteout'
os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_whiteout_folder, exist_ok=True)

# Load the YOLO model
model = YOLO('runs/detect/train20/weights/best.pt')

# Adjustable parameter for white box size
WHITE_BOX_SCALE = 0.5  # Adjust this value to change white box size

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        # Construct full file path
        img_path = os.path.join(input_folder, filename)
        
        # Perform inference
        results = model(img_path)
        
        # Open image with OpenCV for modification
        image = cv2.imread(img_path)
        
        # Create a copy for white box replacement
        whiteout_image = image.copy()
        
        for result in results:
            boxes = result.boxes
            
            for box, cls in zip(boxes.xyxy, boxes.cls):
                if result.names[int(cls)] == 'text_bubble':
                    x1, y1, x2, y2 = map(int, box[:4])
                    # Adjust the box size
                    adj_x1, adj_y1, adj_x2, adj_y2 = adjust_box(x1, y1, x2, y2, WHITE_BOX_SCALE)
                    # Replace the text bubble area with a white box
                    cv2.rectangle(whiteout_image, (adj_x1, adj_y1), (adj_x2, adj_y2), (255, 255, 255), -1)
        
        # Save the original result (with bounding boxes)
        original_result = results[0].plot()
        cv2.imwrite(os.path.join(output_folder, f'result_{filename}'), original_result)
        
        # Save the whiteout result
        cv2.imwrite(os.path.join(output_whiteout_folder, f'whiteout_{filename}'), whiteout_image)
        
        print(f"Processed {filename} - Original saved in {output_folder}, Whiteout saved in {output_whiteout_folder}")

print(f"All images processed. Results saved in folders: {output_folder} and {output_whiteout_folder}")