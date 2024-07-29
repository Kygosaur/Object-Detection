import cv2
import mediapipe as mp

mp_objectron = mp.solutions.objectron
objectron = mp_objectron.Objectron(
    static_image_mode=False,
    max_num_objects=5,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_name='Cup')

cap = cv2.VideoCapture(0)
with mp_objectron.Objectron(static_image_mode=False,
                            max_num_objects=3,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5,
                            model_name='Cup') as objectron:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    # Convert the BGR image to RGB before processing.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = objectron.process(image)

    # Draw the box landmarks on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detected_objects:
      for detected_object in results.detected_objects:
        mp.solutions.drawing_utils.draw_landmarks(
            image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
        cv2.putText(image, f'Index: {detected_object.index}',
                    (int(detected_object.landmarks_2d[0].x), int(detected_object.landmarks_2d[0].y)),
                    cv2.FONT_HERSHEY_SIMPLEX,  1, (0,  255,  0),  2, cv2.LINE_AA)

    cv2.imshow('MediaPipe Objectron', image)
    if cv2.waitKey(1) &   0xFF == ord('q'):
        break

cap.release()
