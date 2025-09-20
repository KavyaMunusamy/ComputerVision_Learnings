import cv2
import numpy as np

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges (you can adjust these)
    colors = {
        "Red": [(0, 120, 70), (10, 255, 255)],
        "Green": [(36, 25, 25), (86, 255, 255)],
        "Blue": [(94, 80, 2), (126, 255, 255)],
        "Yellow": [(15, 150, 150), (35, 255, 255)],
        "Orange": [(5, 150, 150), (15, 255, 255)],
        "White": [(0, 0, 200), (180, 30, 255)],
        "Black": [(0, 0, 0), (180, 255, 30)]
    }


    for name, (lower, upper) in colors.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Find contours of detected color
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:  # filter out small objects
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

    # Show result
    cv2.imshow("Object Detection", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
