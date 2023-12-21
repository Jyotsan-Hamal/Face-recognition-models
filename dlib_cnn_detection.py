import dlib
import cv2

'''
this model is too slow not responding. 
 - slow
 - old 6 years old modelS
'''


# Load the detector
detector = dlib.cnn_face_detection_model_v1('./dlib_model/mmod_human_face_detector.dat')

# Open a connection to the video source (0 for the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video source
    ret, frame = cap.read()

    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    dets = detector(frame_rgb, 1)

    # Loop over the detections
    for i, d in enumerate(dets):
        # Get the bounding box
        x1, y1, x2, y2, score = d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence

        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Display the output frame
    cv2.imshow("Face Detection", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
