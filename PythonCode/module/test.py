from semantic_label_generator import get_semantic_label
import cv2

img = cv2.imread("test.png")
print(img.shape)
cv2.imshow("test1", img)

print(get_semantic_label(img))

cv2.imshow("test",get_semantic_label(img))
cv2.waitKey(0)