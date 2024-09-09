from manga_ocr import MangaOcr
from deep_translator import GoogleTranslator
import os

def extract_text_from_image(image_path):
    """
    Extract text from a manga image using MangaOCR.
    """
    mocr = MangaOcr()
    return mocr(image_path)

def translate_text(text, target_language='en'):
    """
    Translate text using deep_translator's GoogleTranslator.
    """
    translator = GoogleTranslator(source='auto', target=target_language)
    return translator.translate(text)

def process_manga_directory(directory_path, target_language='en'):
    """
    Process all manga images in a directory, extract text, and translate.
    """
    for filename in sorted(os.listdir(directory_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            image_path = os.path.join(directory_path, filename)
            
            # Extract text
            japanese_text = extract_text_from_image(image_path)
            
            # Translate text
            translated_text = translate_text(japanese_text, target_language)
            
            print(f"Image: {filename}")
            print(f"Original (Japanese): {japanese_text}")
            print(f"Translated: {translated_text}")
            print("-" * 50)

# Usage example
if __name__ == "__main__":
    manga_directory = 'testfolder'
    process_manga_directory(manga_directory, target_language='en')



