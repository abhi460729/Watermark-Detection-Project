import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array
import urllib.request
import os

# Download pre-trained SSD MobileNet V2 model from TensorFlow Model Zoo
MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
MODEL_DIR = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
if not os.path.exists(MODEL_DIR):
    os.system(f'wget {MODEL_URL}')
    os.system(f'tar -xzf {MODEL_DIR}.tar.gz')

# Load the saved model
model_path = f'{MODEL_DIR}/saved_model'
model = tf.saved_model.load(model_path)
infer = model.signatures['serving_default']

# Image preprocessing function
def preprocess_image(image_path):
    # Read and resize image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (320, 320))
    # Normalize and convert to tensor
    image_array = img_to_array(image_resized) / 255.0
    image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
    image_tensor = tf.expand_dims(image_tensor, 0)  # Add batch dimension
    return image, image_tensor

# Post-processing function for watermark detection
def postprocess_output(image, output_dict, threshold=0.5):
    boxes = output_dict['detection_boxes'].numpy()[0]
    scores = output_dict['detection_scores'].numpy()[0]
    classes = output_dict['detection_classes'].numpy()[0]
    
    height, width, _ = image.shape
    watermark_detected = False
    for i in range(len(scores)):
        if scores[i] > threshold:
            # Assuming class 1 is watermark (adjust based on your model)
            if int(classes[i]) == 1:
                watermark_detected = True
                ymin, xmin, ymax, xmax = boxes[i]
                # Scale coordinates to original image size
                (left, right, top, bottom) = (xmin * width, xmax * width, ymin * height, ymax * height)
                # Draw bounding box
                cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
                cv2.putText(image, f'Watermark: {scores[i]:.2f}', (int(left), int(top)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image, watermark_detected

# Main function for watermark detection
def detect_watermark(image_path):
    # Preprocess image
    original_image, input_tensor = preprocess_image(image_path)
    # Run inference
    output_dict = infer(input_tensor)
    # Post-process and visualize results
    result_image, watermark_detected = postprocess_output(original_image, output_dict)
    
    # Display result
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Watermark Detection Result')
    plt.show()
    
    return watermark_detected

# Example usage
if __name__ == '__main__':
    # Replace with your image path
    image_path = 'sample_image.jpg'
    if not os.path.exists(image_path):
        print("Please provide a valid image path.")
    else:
        watermark_detected = detect_watermark(image_path)
        print(f"Watermark detected: {watermark_detected}")
