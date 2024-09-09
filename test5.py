import cv2
import numpy as np
from PIL import Image
from manga_ocr import MangaOcr
from ultralytics import YOLO
from deep_translator import GoogleTranslator
import os

# Set up folders
input_folder = 'test_images'
output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)

# Load the YOLO model
yolo_model = YOLO('runs/detect/train20/weights/best.pt')

# Initialize MangaOCR
mocr = MangaOcr()

# Initialize Google Translator from deep_translator
translator = GoogleTranslator(source='ja', target='en')

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        # Construct full file path
        img_path = os.path.join(input_folder, filename)
        
        # Perform YOLO inference
        results = yolo_model(img_path)
        
        # Open image with PIL for MangaOCR
        image = Image.open(img_path)
        
        # Store detected text and translations
        detected_items = []
        
        for result in results:
            boxes = result.boxes
            
            for box, cls in zip(boxes.xyxy, boxes.cls):
                class_name = result.names[int(cls)]
                if class_name in ['text_bubble', 'text_free']:
                    x1, y1, x2, y2 = map(int, box[:4])
                    
                    # Crop the region
                    cropped_region = image.crop((x1, y1, x2, y2))
                    
                    # Perform OCR on the cropped region
                    japanese_text = mocr(cropped_region)
                    
                    # Translate the text
                    try:
                        english_text = translator.translate(japanese_text)
                    except Exception as e:
                        english_text = f"Translation error: {str(e)}"
                    
                    detected_items.append({
                        'class': class_name,
                        'japanese': japanese_text,
                        'english': english_text,
                        'bbox': (x1, y1, x2, y2)
                    })
        
        # Save the results
        output_path = os.path.join(output_folder, f'{filename}_text_translated.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in detected_items:
                f.write(f"Class: {item['class']}\n")
                f.write(f"Japanese: {item['japanese']}\n")
                f.write(f"English: {item['english']}\n")
                f.write(f"Bounding Box: {item['bbox']}\n")
                f.write("\n")
        
        print(f"Processed {filename} - Text extracted, translated, and saved in {output_path}")

print(f"All images processed. Results saved in folder: {output_folder}")