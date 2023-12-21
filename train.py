import cv2
import numpy as np
import os

# Create a list of images and a list of corresponding names
images = []
labels = []
names = []

person = [person for person in os.listdir("dataset")]

# Load the images and labels
for i, person in enumerate(person):
    for image in os.listdir("dataset/" + person):
        images.append(cv2.imread("dataset/" + person + '/' + image, cv2.IMREAD_GRAYSCALE))
        labels.append(i)
    names.append(person)

# Create the LBPH face recognizer
recognizer = cv2.face


# Train the face recognizer
recognizer.train(images, np.array(labels))
