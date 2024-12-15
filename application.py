from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter

app = Flask(__name__)

# Load YOLOv5 pre-trained model
model = YOLO('yolov5x.pt')  # Use YOLOv5x pre-trained model

def detect_and_count(image):
    """
    Detect all objects in an image using YOLOv5 and return the object with the largest count.
    :param image: Input image (numpy array).
    :return: Count of objects detected.
    """
    # Run the model on the image
    results = model(image)

    # Get the class names and predictions
    detected_objects = results[0].boxes.data.numpy()
    class_names = model.names

    if len(detected_objects) == 0:
        return 0  # No objects detected

    # Extract class IDs and corresponding class names from the detections
    class_ids = [int(obj[5]) for obj in detected_objects]

    # Count the occurrences of each object class
    class_count = Counter(class_ids)

    # Identify the class ID with the highest count
    most_common_class_id, count = class_count.most_common(1)[0]
    return count

@app.route('/detect', methods=['POST'])
def detect():
    """
    Handle image upload and object detection, then return the count in JSON format.
    :return: JSON response with the object count.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read the uploaded image as a numpy array
    img_array = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({'error': 'Invalid image file'}), 400

    # Detect and count objects
    object_count = detect_and_count(image)

    # Return the result as JSON
    return jsonify({'object_count': object_count})

if __name__ == '__main__':
    app.run(debug=True)