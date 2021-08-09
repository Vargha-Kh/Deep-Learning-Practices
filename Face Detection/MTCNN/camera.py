import cv2
import mtcnn
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

face_detector = mtcnn.MTCNN()
video = cv2.VideoCapture(0)

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        print("null")
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_detector.detect_faces(frame_rgb)
    for res in result:
        x1, y1, width, height = res['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        confidence = res['confidence']
        key_points = res['keypoints'].values()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f'conf: {confidence:.3f}', (x1, y1), cv2.FONT_ITALIC, 1, (0, 0, 255), 1)
        for points in key_points:
            cv2.circle(frame, points, 5, (0, 255, 0), thickness=-1)
    cv2.imshow('WhatsApp Image 2021-05-26 at 5.52.04 PM', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
