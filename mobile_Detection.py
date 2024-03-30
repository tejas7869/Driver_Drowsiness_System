import cv2
import torch
import winsound

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Get bounding box coordinates for mobile phones
    bboxes = results.xyxy[0]
    for bbox in bboxes:
        class_id = int(bbox[5])
        if class_id == 67:  # Class ID for "cell phone" in COCO dataset
            x1, y1, x2, y2 = map(int, bbox[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, 'Mobile Phone', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Beep alarm
            winsound.Beep(1000, 500)  # Change frequency (1000) and duration (500) as needed

    # Display the frame
    cv2.imshow('Mobile Phone Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
cap.release()
cv2.destroyAllWindows()
