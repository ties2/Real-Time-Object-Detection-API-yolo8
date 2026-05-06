import cv2  # Works with images and videos  
import torch  # Helps with AI models  
import os  # Helps with file paths  
import numpy as np  # Works with numbers and arrays  
from ultralytics import YOLO  # The main YOLOv8 object detection model  
from torchvision import transforms  # Helps convert images for AI  
from PIL import Image  # Helps open images  

# Set up paths
data_path = "./yolo8/data/images"
model_path = "./models/yolov8.pt"
output_path = "./output"
os.makedirs(output_path, exist_ok=True)

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Using the pre-trained YOLOv8 nano model

# Preprocessing function
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB") # Open image
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Resize image to 640x640 pixels
        transforms.ToTensor()  # Convert image to numbers
    ])
    return transform(image).unsqueeze(0) # Add extra dimension (needed for AI)

# Inference function
def detect_objects(image_path):
    results = model(image_path) # Send image to the AI model
    return results  # Return detection results

# Process all images in dataset
for img_file in os.listdir(data_path):  # Look at every image in the folder
    img_path = os.path.join(data_path, img_file)  # Get full image path
    results = detect_objects(img_path) # Detect objects
    
    # Draw bounding boxes
    img = cv2.imread(img_path)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box position
            conf = box.conf[0].item() # Get confidence (how sure AI is)
            label = model.names[int(box.cls[0].item())]   # Get object name
            
            # Draw the box on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Put object name + confidence score
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save output
    output_img_path = os.path.join(output_path, img_file)  # Set save path
    cv2.imwrite(output_img_path, img) # Save image with boxes

print("Object detection complete! Check the output folder.")
