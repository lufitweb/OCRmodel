import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from manga_ocr import MangaOcr
from deep_translator import GoogleTranslator
import os
from ultralytics import YOLO

# Set up folders
input_folder = 'test_images'
output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)

# Load the YOLO model
model = YOLO('runs/detect/train20/weights/best.pt')

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        # Construct full file path
        img_path = os.path.join(input_folder, filename)
        
        # Perform inference
        results = model(img_path)
        
        # Process and save results
        for i, result in enumerate(results):
            # Get the original image
            original_image = Image.open(img_path)
            
            # Plot the results on the image
            result_plotted = result.plot()
            result_image = Image.fromarray(result_plotted[..., ::-1])  # RGB PIL Image
            
            # Construct output filename
            output_filename = f'result_{filename.split(".")[0]}_{i+1}.jpg'
            output_path = os.path.join(output_folder, output_filename)
            
            # Save the result image
            result_image.save(output_path)
            
            print(f"Processed {filename} - Saved as {output_filename}")

print(f"All images processed. Results saved in folder: {output_folder}")
        



