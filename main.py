from flask import Flask, request, send_file, jsonify
from PIL import Image, ExifTags
import io
from ultralytics import YOLO
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

app = Flask(__name__)

# Load the custom-trained YOLOv8s model
model = YOLO('YOLOv8s_e100_lr0.0001_32.pt')

# Define the classes
class_names = [
    'Botol Kaca',
    'Botol Plastik',
    'Galon',
    'Gelas Plastik',
    'Kaleng',
    'Kantong Plastik',
    'Kantong Semen',
    'Kardus',
    'Kemasan Plastik',
    'Kertas Bekas',
    'Koran',
    'Pecahan Kaca',
    'Toples Kaca',
    'Tutup Galon'
]

def correct_image_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        exif = image._getexif()
        if exif is not None:
            orientation = exif.get(orientation, 1)
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        pass

    return image

def get_prediction(image_bytes, threshold=0.5):
    image = Image.open(image_bytes).convert("RGB")
    results = model(image)
    boxes = []
    labels = []
    scores = []

    for result in results[0].boxes:  # Iterate through detected objects
        x1, y1, x2, y2 = result.xyxy[0].cpu().numpy()
        score = result.conf[0].cpu().numpy()
        label = class_names[int(result.cls[0].cpu().numpy())]
        if score > threshold:  # Only keep if the score is above a threshold
            boxes.append([x1, y1, x2, y2])
            labels.append(label)
            scores.append(score)

    return image, boxes, labels, scores

def draw_boxes(image, boxes, labels, scores):
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1, f'{label} : {score:.2f}', bbox=dict(facecolor='yellow', alpha=0.5))

    plt.axis('off')

    # Save the image with boxes to a BytesIO object
    img_io = io.BytesIO()
    fig.savefig(img_io, format='png')
    plt.close(fig)
    img_io.seek(0)
    return Image.open(img_io)

@app.route("/")
def main():
    return "Response Successful!"

@app.route('/image', methods=['POST'])
def image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'}), 400

    image_file = request.files['image']
    image_bytes = io.BytesIO(image_file.read())
    image_format = Image.open(image_bytes).format  # Detect image format
    image_bytes.seek(0)  # Reset the stream to the beginning

    # Correct image orientation
    image = Image.open(image_bytes)
    image = correct_image_orientation(image)

    # Resize the image to 750x1000 pixels
    image = image.resize((750, 1000))
    image_bytes = io.BytesIO()
    image.save(image_bytes, format=image_format)
    image_bytes.seek(0)  # Reset the stream to the beginning

    # Get predictions
    img, boxes, labels, scores = get_prediction(image_bytes, threshold=0.5)
    img_with_boxes = draw_boxes(img, boxes, labels, scores)

    # Convert to RGB if saving as JPEG
    if image_format.upper() == 'JPEG':
        img_with_boxes = img_with_boxes.convert("RGB")

    # Save the image with boxes to a BytesIO object
    img_io = io.BytesIO()
    img_with_boxes.save(img_io, format=image_format)  # Save in the original format
    img_io.seek(0)

    return send_file(img_io, mimetype=f'image/{image_format.lower()}')

@app.route('/text', methods=['POST'])
def text():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'}), 400

    image_file = request.files['image']
    image_bytes = io.BytesIO(image_file.read())
    image_format = Image.open(image_bytes).format  # Detect image format
    image_bytes.seek(0)  # Reset the stream to the beginning

    # Correct image orientation
    image = Image.open(image_bytes)
    image = correct_image_orientation(image)

    # Resize the image to 750x1000 pixels
    image = image.resize((750, 1000))
    image_bytes = io.BytesIO()
    image.save(image_bytes, format=image_format)
    image_bytes.seek(0)  # Reset the stream to the beginning

    # Get predictions
    img, boxes, labels, scores = get_prediction(image_bytes, threshold=0.5)
    img_with_boxes = draw_boxes(img, boxes, labels, scores)

    # Convert to RGB if saving as JPEG
    if image_format.upper() == 'JPEG':
        img_with_boxes = img_with_boxes.convert("RGB")

    # Prepare the JSON response
    predictions = []
    for box, label, score in zip(boxes, labels, scores):
        predictions.append({
            'label': label,
            'score': float(score),  # Convert numpy.float32 to float
            'box': [int(b) for b in box]  # Convert numpy.float32 to list of int
        })

    response = {
        'predictions': predictions
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
