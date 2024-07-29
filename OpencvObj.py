import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3,  640)  # Width
cap.set(4,  480)  # Height

while True:
    #webcam input
    success, img = cap.read()

    # convert to grayscale 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Perform object detection using Haar cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,  30))

    # draw a bounding box around face detected
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0,  255,  0),  3)

    # Display the result
    cv2.imshow('Webcam', img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) &  0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
