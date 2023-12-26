import cv2
import numpy as np
import time

# Load the model
net = cv2.dnn.readNetFromCaffe('../dnn_model/deploy.prototxt.txt', '../dnn_model/res10_300x300_ssd_iter_140000.caffemodel')

# Set preferable backend and target to CUDA
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Open the video file
cap = cv2.VideoCapture('./videos/cctv.mp4')

# Get the frames per second (FPS) of the video
fps = cap.get(cv2.CAP_PROP_FPS)

while True:
    # Record the start time to calculate FPS
    start_time = time.time()

    # Read a frame from the video source
    ret, frame = cap.read()

    # Break the loop if the video is finished
    if not ret:
        break

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
        if confidence > 0.2:
            # Compute the coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the bounding box
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

    # Calculate and display FPS
    current_time = time.time()
    elapsed_time = current_time - start_time

    # Check if elapsed_time is greater than zero to avoid division by zero
    if elapsed_time > 0:
        fps_text = f"FPS: {1 / elapsed_time:.2f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the output frame
    cv2.imshow("Face Detection", frame)

    # Introduce a delay to control the playback speed
    delay = int(350 / fps)  # Delay in milliseconds
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
