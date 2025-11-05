from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
import pytesseract
from PIL import Image

app = Flask(__name__)

# Configure upload and result folders
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Set the path to Tesseract OCR (update this path if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def detect_plate(image_path):
    """Detects the license plate in the image."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur and edge detection
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area (descending)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for contour in contours:
        # Approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # Check if the contour is a quadrilateral (likely a license plate)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            plate_img = img[y:y+h, x:x+w]
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            return img, plate_img

    return img, None

def extract_text(image):
    """Extracts text from the license plate image."""
    if image is None:
        return "No plate detected."

    # Convert OpenCV BGR image to RGB for PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    # Use Tesseract to extract text
    custom_config = r'--oem 3 --psm 7'
    text = pytesseract.image_to_string(pil_image, config=custom_config)
    return text.strip()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            # Create directories if they don't exist
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

            # Save the uploaded file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Detect plate
            original_img, plate_img = detect_plate(filepath)

            # Save results
            original_filename = 'original_' + file.filename
            plate_filename = 'plate_' + file.filename
            original_path = os.path.join(app.config['RESULT_FOLDER'], original_filename)
            plate_path = os.path.join(app.config['RESULT_FOLDER'], plate_filename)
            cv2.imwrite(original_path, original_img)

            if plate_img is not None:
                cv2.imwrite(plate_path, plate_img)
                plate_text = extract_text(plate_img)
            else:
                plate_text = "No plate detected."

            return render_template(
                'index.html',
                original=original_filename,
                plate=plate_filename,
                text=plate_text
            )

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
