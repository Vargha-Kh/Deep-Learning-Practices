import cv2

camera_index = 0
video = cv2.VideoCapture(camera_index)
detector = cv2.CascadeClassifier("/home/vargha/Desktop/haarcascade_frontalface_default.xml")

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        print("null")
        break
    boxes = detector.detectMultiScale(frame)
    for box in boxes:
        x1, y1, width, height = box
        x2, y2 = x1 + width, y1 + height
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('WhatsApp Image 2021-05-26 at 5.52.04 PM', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
