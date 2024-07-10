import numpy as np
import cv2
from masking import label_to_points

def extract(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    image_height, image_weight = image.shape[:2]
    scaled_points = (points * [image_weight, image_height]).astype(np.float32)

    plate_h, plate_w = 100, 450
    image_plate_coordinates = np.array([[0, 0], [plate_w - 1, 0], [plate_w - 1, plate_h - 1], [0, plate_h - 1]],
                                       dtype=np.float32)

    homo_matrix, _ = cv2.findHomography(scaled_points, image_plate_coordinates)
    warped_cover = cv2.warpPerspective(image, homo_matrix, (plate_w, plate_h))

    return warped_cover


# MAIN
if __name__ == '__main__':
    img = cv2.imread("../resource/image/day_02553.jpg")
    cover = cv2.imread("../resource/kntu.jpg")

    with open("../resource/label/6e844fd3-day_02553.txt", 'r') as file:
        points = label_to_points(file.read().strip().split(" "))

    cv2.imshow("img", extract(img, points))

    cv2.waitKey()
