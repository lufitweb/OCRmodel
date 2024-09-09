import cv2
import numpy as np
from ultralytics import YOLO
import os

def precise_whiteout(image, x1, y1, x2, y2):
    # Extract the region of interest (ROI)
    roi = image[y1:y2, x1:x2]
    
    # Convert ROI to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour (assuming it's the text bubble)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create a mask for the contour
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [largest_contour], 0, (255), -1)
        
        # Apply the mask to create white fill
        roi[mask == 255] = [255, 255, 255]
    
    # Put the modified ROI back into the image
    image[y1:y2, x1:x2] = roi
    return image

# Set up folders
input_folder = 'test_images'
output_folder = 'output'
output_whiteout_folder = 'output_whiteout'
os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_whiteout_folder, exist_ok=True)

# Load the YOLO model
model = YOLO('runs/detect/train20/weights/best.pt')

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        img_path = os.path.join(input_folder, filename)
        
        # Perform inference
        results = model(img_path)
        
        # Open image with OpenCV for modification
        image = cv2.imread(img_path)
        whiteout_image = image.copy()
        
        for result in results:
            boxes = result.boxes
            
            for box, cls in zip(boxes.xyxy, boxes.cls):
                if result.names[int(cls)] == 'text_bubble':
                    x1, y1, x2, y2 = map(int, box[:4])
                    whiteout_image = precise_whiteout(whiteout_image, x1, y1, x2, y2)
        
        # Save the original result (with bounding boxes)
        original_result = results[0].plot()
        cv2.imwrite(os.path.join(output_folder, f'result_{filename}'), original_result)
        
        # Save the whiteout result
        cv2.imwrite(os.path.join(output_whiteout_folder, f'whiteout_{filename}'), whiteout_image)
        
        print(f"Processed {filename} - Original saved in {output_folder}, Whiteout saved in {output_whiteout_folder}")

print(f"All images processed. Results saved in folders: {output_folder} and {output_whiteout_folder}")