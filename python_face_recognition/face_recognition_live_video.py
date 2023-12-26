import face_recognition
import cv2
import numpy as np
# Load a sample picture and learn how to recognize it.
jyotsan_image = face_recognition.load_image_file("dataset/jyotsan/image2.jpg")
jyotsan_face_encoding = face_recognition.face_encodings(jyotsan_image)[0]

# Load a second sample picture and learn how to recognize it.
tom_image = face_recognition.load_image_file("dataset/tom/image1.jpg")
tom_face_encoding = face_recognition.face_encodings(tom_image)[0]

# Load a second sample picture and learn how to recognize it.
angelina_image = face_recognition.load_image_file("dataset/angelina/image2.jpg")
angelina_face_encoding = face_recognition.face_encodings(angelina_image)[0]

# Load a second sample picture and learn how to recognize it.
yukesh_image = face_recognition.load_image_file("dataset/yukesh/image2.jpg")
yukesh_face_encoding = face_recognition.face_encodings(yukesh_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    jyotsan_face_encoding,
    tom_face_encoding,
    angelina_face_encoding,
    yukesh_face_encoding
    
]
known_face_names = [
    "Jyotsan",
    "tom",
    "angelina",
    "yukesh"
]


net = cv2.dnn.readNetFromCaffe('./dnn_model/deploy.prototxt.txt', './dnn_model/res10_300x300_ssd_iter_140000.caffemodel')
# Set preferable backend and target to CUDA
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []

def get_face_locations(rgb_small_frame):
    # Get the height and width of the frame
    (h, w) = rgb_small_frame.shape[:2]

    # Create a blob from the image and perform forward pass
    blob = cv2.dnn.blobFromImage(rgb_small_frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # Initialize list to store face locations
    face_locations = []

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Check if confidence meets a certain threshold (adjust as needed)
        if confidence > 0.3:
            # Compute the coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Add the face location to the list
            face_locations.append((startY, endX, endY, startX))  # Format: (top, right, bottom, left)
    
    return face_locations,confidence

def get_faces(frame):

    # Resize frame of video to 1/10 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.1, fy=0.1)
    
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_image = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Find all the faces and face encodings in the current frame of video
    face_locations,confidence = get_face_locations(rgb_image)
    
    # face_locations = face_recognition.face_locations(frame)
    # print(f"face locations : {face_locations}")
    # print(f"type(face_locations) = {type(face_locations)}")
    # print(f"face locations : {len(face_locations)}")
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
   
    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Person"

        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        
        face_names.append(name)


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/10 size
        top *= 10
        right *= 10
        bottom *= 10
        left *= 10

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (255,0,0), 1)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name+f"{confidence}", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    return frame


video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
   
    # Apply face recognition to the frame
    processed_frame = get_faces(frame)
    
    # Display the resulting frame
    cv2.imshow('Video', processed_frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()