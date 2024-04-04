import cv2


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0) 


prev_centroids = []

while True:
   
    ret, frame = cap.read()
    if not ret:
        break

  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

   
    curr_centroids = []
    for (x, y, w, h) in faces:
        centroid_x = x + w // 2
        centroid_y = y + h // 2
        curr_centroids.append((centroid_x, centroid_y))


        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    for centroid in curr_centroids:
        if centroid not in prev_centroids:
         
            pass
     
        cv2.circle(frame, centroid, 4, (0, 255, 0), -1)


    cv2.imshow('Face Tracking', frame)


    prev_centroids = curr_centroids[:]


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
