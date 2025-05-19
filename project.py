import cv2
import numpy as np
image = cv2.imread('defect_input.jpg')  
if image is None:
    raise FileNotFoundError("Image not found. Check the path.")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
kernel = np.ones((3, 3), np.uint8)
cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    if area > 200:  
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [cnt], -1, 255, -1)  
        if area < 500:
            label = "Crack"
        elif w > h:
            label = "Hole"
        else:
            label = "Stain"
        cv2.putText(image, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
cv2.imshow("Detected Defects", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("defect_output.jpg", image)