import cv2
import numpy as np
from PIL import Image
import sys
import math
import os

def Distance(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx * dx + dy * dy)

def ScaleRotateTranslate(image, angle, center=None, new_center=None, scale=None, resample=Image.BICUBIC):
    if (scale is None) and (center is None):
        return image.rotate(angle=angle, resample=resample)
    nx, ny = x, y = center
    sx = sy = 1.0
    if new_center:
        (nx, ny) = new_center
    if scale:
        (sx, sy) = (scale, scale)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    a = cosine / sx
    b = sine / sx
    c = x - nx * a - ny * b
    d = -sine / sy
    e = cosine / sy
    f = y - nx * d - ny * e
    return image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f), resample=resample)

def CropFace(image, eye_left=(0, 0), eye_right=(0, 0), offset_pct=(0.2, 0.2), dest_sz=(70, 70)):
    offset_h = math.floor(float(offset_pct[0]) * dest_sz[0])
    offset_v = math.floor(float(offset_pct[1]) * dest_sz[1])
    
    eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
    rotation = -math.atan2(float(eye_direction[1]), float(eye_direction[0]))
    
    dist = Distance(eye_left, eye_right)
    reference = dest_sz[0] - 2.0 * offset_h
    scale = float(dist) / float(reference)
    
    image = ScaleRotateTranslate(image, center=eye_left, angle=rotation)
    
    crop_xy = (eye_left[0] - scale * offset_h, eye_left[1] - scale * offset_v)
    crop_size = (dest_sz[0] * scale, dest_sz[1] * scale)
    image = image.crop((int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0] + crop_size[0]), int(crop_xy[1] + crop_size[1])))
    
    image = image.resize(dest_sz, Image.ANTIALIAS)
    return image

# Load the model
net = cv2.dnn.readNetFromCaffe('./dnn_model/deploy.prototxt.txt', './dnn_model/res10_300x300_ssd_iter_140000.caffemodel')

# Path to the dataset folder
dataset_folder = 'dataset'
output_folder = 'dataset2'

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through images in the dataset folder
for person_folder in os.listdir(dataset_folder):
    person_folder_path = os.path.join(dataset_folder, person_folder)
    output_person_folder = os.path.join(output_folder, person_folder)
    
    # Create output person folder if it doesn't exist
    if not os.path.exists(output_person_folder):
        os.makedirs(output_person_folder)
    
    for image_name in os.listdir(person_folder_path):
        image_path = os.path.join(person_folder_path, image_name)
        
        # Read the image
        frame = cv2.imread(image_path)

        # Convert the image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create blob from frame
        blob = cv2.dnn.blobFromImage(cv2.resize(frame_rgb, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        # Set the blob as input and obtain the face detections
        net.setInput(blob)
        detections = net.forward()

        # Loop over the detections
        for i in range(0, detections.shape[2]):
            # Extract the confidence associated with the prediction
            confidence = detections[0, 0, i, 2]

            # Filter out weak detections
            if confidence > 0.5:
                # Compute the coordinates of the bounding box
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw the bounding box
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

                # Crop the detected face
                detected_face = frame[startY:endY, startX:endX]

                # Save the cropped face to the output folder
                output_image_path = os.path.join(output_person_folder, f"{image_name.split('.')[0]}_face.jpg")
                cv2.imwrite(output_image_path, detected_face)
