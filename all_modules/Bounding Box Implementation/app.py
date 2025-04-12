from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import supervision as sv
from inference_sdk import InferenceHTTPClient
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Roboflow Inference Client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="F6VAefqHnSD2MQyEyV2w"  # Replace with your actual API key
)

# Reference height and scaling factor for timer calculation
REFERENCE_HEIGHT = 200  # Adjust as needed
SCALING_FACTOR = 10  # Adjust as needed

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'images' not in request.files:
        return jsonify({'error': 'No file part'})
    
    files = request.files.getlist('images')
    results = []

    for file in files:
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filename)
        
        # Load Image
        image = cv2.imread(filename)
        if image is None:
            return jsonify({'error': 'Failed to load image'})

        try:
            result = CLIENT.infer(filename, model_id="vip-tzvbu/2")  # Ensure correct model_id
            detections = sv.Detections.from_inference(result)
        except Exception as e:
            return jsonify({'error': f'Inference error: {str(e)}'})
        
        for detection in result.get("predictions", []):
            x, y, w, h = map(int, [detection["x"], detection["y"], detection["width"], detection["height"]])

            # Convert center coordinates to top-left and bottom-right
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            # Calculate Timer in seconds
            timer_seconds = round((h * SCALING_FACTOR) / REFERENCE_HEIGHT, 2)

            # Draw the corrected bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"H:{h}px, Timer:{timer_seconds}s", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        processed_filename = os.path.join(UPLOAD_FOLDER, 'processed_' + file.filename)
        cv2.imwrite(processed_filename, image)
        
        results.append({'image': processed_filename, 'height': h, 'timer': timer_seconds})
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
