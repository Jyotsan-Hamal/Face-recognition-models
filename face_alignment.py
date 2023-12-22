import cv2
import dlib
import numpy as np

# Load the model
net = cv2.dnn.readNetFromCaffe('./dnn_model/deploy.prototxt.txt', './dnn_model/res10_300x300_ssd_iter_140000.caffemodel')

# Load the facial landmarks predictor
predictor = dlib.shape_predictor('./face_Alignment_model/shape_predictor_68_face_landmarks_GTX.dat')

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

            # Get the landmarks prediction
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            landmarks = predictor(gray, dlib.rectangle(left=startX, top=startY, right=endX, bottom=endY))

            # Get the coordinates of the left and right eye landmarks
            left_eye = np.mean(np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]), axis=0)
            right_eye = np.mean(np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]), axis=0)

            # Compute the angle between the eye landmarks
            angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))

            # Rotate the image to align the eyes
            M = cv2.getRotationMatrix2D((frame.shape[1]//2, frame.shape[0]//2), angle, 1)
            frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

    # Display the output frame
    cv2.imshow("Face Detection", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
