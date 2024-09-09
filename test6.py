import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from manga_ocr import MangaOcr
from ultralytics import YOLO
from deep_translator import GoogleTranslator
import os
import textwrap

# Set up folders
input_folder = 'test_images'
output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)

# Load the YOLO model
yolo_model = YOLO('runs/detect/train20/weights/best.pt')

# Initialize MangaOCR
mocr = MangaOcr()

# Initialize Google Translator
translator = GoogleTranslator(source='ja', target='en')

def get_font_size(text, font_path, image_fraction, max_font_size, box_width, box_height):
    font_size = 1
    font = ImageFont.truetype(font_path, font_size)
    image = Image.new('RGB', (box_width, box_height))
    draw = ImageDraw.Draw(image)
    
    while True:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        if text_width < box_width * image_fraction and text_height < box_height * image_fraction:
            font_size += 1
            font = ImageFont.truetype(font_path, font_size)
        else:
            break
    
    return min(font_size - 1, max_font_size)

def add_text_to_image(image, text, box, font_path="font/CC WILD WORDS ITALIC.TTF", font_color=(0, 0, 0), max_font_size=30):
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    
    # Calculate the appropriate font size
    font_size = get_font_size(text, font_path, 0.9, max_font_size, box_width, box_height)
    font = ImageFont.truetype(font_path, font_size)
    
    # Wrap the text
    avg_char_width = sum(draw.textbbox((0, 0), char, font=font)[2] for char in text) / len(text)
    max_chars_per_line = int(box_width / avg_char_width)
    wrapped_text = textwrap.fill(text, width=max_chars_per_line)
    
    # Calculate text position for center alignment
    bbox = draw.textbbox((0, 0), wrapped_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = x1 + (box_width - text_width) / 2
    text_y = y1 + (box_height - text_height) / 2
    
    # Draw the text
    draw.multiline_text((text_x, text_y), wrapped_text, font=font, fill=font_color, align='center')
    
    return image

def add_text_to_image(image, text, box, font_path="font/CC WILD WORDS ITALIC.TTF", font_color=(0, 0, 0), max_font_size=30):
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    
    # Calculate the appropriate font size
    font_size = get_font_size(text, font_path, 0.9, max_font_size, box_width, box_height)
    font = ImageFont.truetype(font_path, font_size)
    
    # Wrap the text
    avg_char_width = sum(draw.textbbox((0, 0), char, font=font)[2] for char in text) / len(text)
    max_chars_per_line = int(box_width / avg_char_width)
    wrapped_text = textwrap.fill(text, width=max_chars_per_line)
    
    # Calculate text position for center alignment
    bbox = draw.textbbox((0, 0), wrapped_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = x1 + (box_width - text_width) / 2
    text_y = y1 + (box_height - text_height) / 2
    
    # Draw the text
    draw.multiline_text((text_x, text_y), wrapped_text, font=font, fill=font_color, align='center')
    
    return image

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        # Construct full file path
        img_path = os.path.join(input_folder, filename)

        # Perform YOLO inference
        results = yolo_model(img_path)

        # Open image with PIL for processing
        image = Image.open(img_path)
        
        # Create a copy of the image for drawing
        draw_image = image.copy()

        # Store detected text and translations
        detected_items = []

        for result in results:
            boxes = result.boxes
            for box, cls in zip(boxes.xyxy, boxes.cls):
                class_name = result.names[int(cls)]
                if class_name in ['text_bubble', 'text_free']:
                    x1, y1, x2, y2 = map(int, box[:4])

                    # Create a white rectangle
                    draw = ImageDraw.Draw(draw_image)
                    draw.rectangle([x1, y1, x2, y2], fill="white")

                    # Crop the region from the original image
                    cropped_region = image.crop((x1, y1, x2, y2))

                    # Perform OCR on the cropped region
                    japanese_text = mocr(cropped_region)

                    # Translate the text
                    try:
                        english_text = translator.translate(japanese_text)
                    except Exception as e:
                        english_text = f"Translation error: {str(e)}"

                    # Add translated text to the image
                    draw_image = add_text_to_image(draw_image, english_text, (x1, y1, x2, y2))

                    detected_items.append({
                        'class': class_name,
                        'japanese': japanese_text,
                        'english': english_text,
                        'bbox': (x1, y1, x2, y2)
                    })

        # Save the processed image
        output_image_path = os.path.join(output_folder, f'{filename}_translated.png')
        draw_image.save(output_image_path)

        # Save the text results
        output_text_path = os.path.join(output_folder, f'{filename}_text_translated.txt')
        with open(output_text_path, 'w', encoding='utf-8') as f:
            for item in detected_items:
                f.write(f"Class: {item['class']}\n")
                f.write(f"Japanese: {item['japanese']}\n")
                f.write(f"English: {item['english']}\n")
                f.write(f"Bounding Box: {item['bbox']}\n")
                f.write("\n")

        print(f"Processed {filename} - Text extracted, translated, and overlaid. Saved as {output_image_path}")
        print(f"Text results saved in {output_text_path}")

print(f"All images processed. Results saved in folder: {output_folder}")