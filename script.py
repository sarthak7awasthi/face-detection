import cv2

# Initialize face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize video capture (use 0 for default webcam)
cap = cv2.VideoCapture(0)

# Constants for tracking
MOVEMENT_THRESHOLD = 10  # Adjust as needed
THRESHOLD_X = 20  # Adjust for deadzone around center (optional)
THRESHOLD_Y = 20  # Adjust for deadzone around center (optional)

# Initialize previous centroid
prev_centroid = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Calculate centroids for each detected face
    curr_centroids = []
    for (x, y, w, h) in faces:
        centroid_x = x + w // 2
        centroid_y = y + h // 2
        curr_centroids.append((centroid_x, centroid_y))

        # Draw bounding boxes around detected faces
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Track faces using centroid tracking
    if curr_centroids:
        if prev_centroid:
            offset_x = curr_centroids[0][0] - prev_centroid[0]
            offset_y = curr_centroids[0][1] - prev_centroid[1]

            # Analyze offset for guidance (consider deadzone around center)
            if abs(offset_x) > MOVEMENT_THRESHOLD and abs(offset_x) > THRESHOLD_X:
                direction_x = "Move right" if offset_x > 0 else "Move left"
                print(f"Move {direction_x} to keep face in frame.")
            if abs(offset_y) > MOVEMENT_THRESHOLD and abs(offset_y) > THRESHOLD_Y:
                direction_y = "Move down" if offset_y > 0 else "Move up"
                print(f"Move {direction_y} to keep face in frame.")

        # Update previous centroid
        prev_centroid = curr_centroids[0]

    # Display the resulting frame
    cv2.imshow('Face Tracking with Lock-on', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture
cap.release()
cv2.destroyAllWindows()
