import pytesseract
from PIL import Image
import cv2

def extract_text(image):
    # Convert OpenCV BGR image to RGB for PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    # Use Tesseract to extract text
    custom_config = r'--oem 3 --psm 7'
    text = pytesseract.image_to_string(pil_image, config=custom_config)
    return text.strip()
