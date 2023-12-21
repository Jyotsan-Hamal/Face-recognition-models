import cv2
import numpy as np

'''
This model works gives the accuracy of 90%
benefits
- effective against the face postion
- 
'''

# Load the model
net = cv2.dnn.readNetFromCaffe('./dnn_model/deploy.prototxt.txt', './dnn_model/res10_300x300_ssd_iter_140000.caffemodel')

# Open a connection to the video source (0 for the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video source
    ret, frame = cap.read()

    # Create blob from frame
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

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

    # Display the output frame
    cv2.imshow("Face Detection", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
