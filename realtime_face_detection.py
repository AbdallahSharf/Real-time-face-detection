from ultralytics import YOLO
import cv2

model = YOLO(r"E:/4th Year/AI model/runs/detect/train/weights/best.pt")

results = model.predict(source=0, stream=True, conf=0.4)

for result in results:
    frame = result.orig_img
    boxes = result.boxes

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()