import dlib
import cv2
# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    # Resize frame of video to 1/10 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.04, fy=0.04)
    
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_image = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    detected_faces = face_detector(rgb_image, 1)
    for i, face_rect in enumerate(detected_faces):
        print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))


        # Draw a box around the face
        cv2.rectangle(frame, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (255,0,0), 1)
        # Draw a label with a name below the face
        # cv2.rectangle(frame, (face_rect.left(), face_rect.bottom() - 35), (face_rect.right(), face_rect.bottom()), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
    # Display the output frame
    cv2.imshow("Face Detection", frame)
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()

