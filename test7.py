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

def get_text_size(text, font):
    dummy_image = Image.new('RGB', (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_image)
    return dummy_draw.textbbox((0, 0), text, font=font)

def get_font_size(text, font_path, max_width, max_height, min_font_size=10, max_font_size=40):
    font_size = max_font_size
    font = ImageFont.truetype(font_path, font_size)
    text_bbox = get_text_size(text, font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    while (text_width > max_width or text_height > max_height) and font_size > min_font_size:
        font_size -= 1
        font = ImageFont.truetype(font_path, font_size)
        text_bbox = get_text_size(text, font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
    
    return font_size

def add_text_to_image(image, text, box, font_path="font/CC WILD WORDS ITALIC.TTF", font_color=(0, 0, 0), padding=5):
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    
    # Calculate the maximum width and height for the text
    max_text_width = box_width - (padding * 2)
    max_text_height = box_height - (padding * 2)
    
    # Get the appropriate font size
    font_size = get_font_size(text, font_path, max_text_width, max_text_height)
    font = ImageFont.truetype(font_path, font_size)
    
    # Wrap the text
    avg_char_width = sum(draw.textbbox((0, 0), char, font=font)[2] for char in text) / len(text)
    max_chars_per_line = int(max_text_width / avg_char_width)
    wrapped_text = textwrap.fill(text, width=max_chars_per_line)
    
    # Calculate text size
    text_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Calculate white layer size and position
    white_layer_width = text_width + (padding * 2)
    white_layer_height = text_height + (padding * 2)
    white_layer_x = x1 + (box_width - white_layer_width) // 2
    white_layer_y = y1 + (box_height - white_layer_height) // 2
    
    # Draw white layer
    draw.rectangle([white_layer_x, white_layer_y, white_layer_x + white_layer_width, white_layer_y + white_layer_height], fill="white")
    
    # Calculate text position
    text_x = white_layer_x + padding
    text_y = white_layer_y + padding
    
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