import cv2
from mtcnn.mtcnn import MTCNN


#Too accurate if we don't resize the image 1/10. otherwise it detects face same as resnet dnn model


def detect_faces_in_video(video_path):
    # Load the MTCNN model
    detector = MTCNN()

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()

        if not ret:
            break
        # Resize frame of video to 1/10 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.1, fy=0.1)
        
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_image = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        # Detect faces in the current frame
        faces = detector.detect_faces(rgb_image)

        # Draw rectangles around the faces
        for face in faces:
            x, y, width, height = face['box']
            
            # Multiply the top, bottom, left, and right by the inverse of the scaling factor (10 in this example)
            x = int(x * 10)
            y = int(y * 10)
            width = int(width * 10)
            height = int(height * 10)

            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 1)

        # Display the frame with face rectangles
        cv2.imshow('Video with Face Detection', frame)

        # Exit the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the window
    cap.release()
    cv2.destroyAllWindows()

# Provide the path to the video file
video_path = '../videos/detect.mp4'

# Call the function to detect faces in the video
detect_faces_in_video(video_path)
