import numpy as np
import cv2


def label_to_points(label):
    if len(label) != 9:
        raise ValueError("Label must have 9 values representing class and four corner points.")

    return np.array(label[1:]).reshape((4, 2)).astype(np.float32)


def mask(image: np.ndarray, points: np.ndarray, cover: np.ndarray) -> np.ndarray:
    image_height, image_weight = image.shape[:2]
    scaled_points = (points * [image_weight, image_height]).astype(np.float32)

    cover_h, cover_w = cover.shape[:2]
    cover_corner_coordinates = np.array([[0, 0], [cover_w - 1, 0], [cover_w - 1, cover_h - 1], [0, cover_h - 1]],
                                        dtype=np.float32)

    homo_matrix, _ = cv2.findHomography(cover_corner_coordinates, scaled_points)
    warped_cover = cv2.warpPerspective(cover, homo_matrix, (image_weight, image_height))

    mask = (warped_cover > 0).any(axis=2).astype(np.uint8) * 255

    result = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
    result += cv2.bitwise_and(warped_cover, warped_cover, mask=mask)

    return result


# MAIN
img = cv2.imread("../resource/image/night (1782).jpg")
cover = cv2.imread("../resource/kntu.jpg")

with open("../resource/label/598e99ea-night_1782.txt", 'r') as file:
    points = label_to_points(file.read().strip().split(" "))

cv2.imshow("img", mask(img, points, cover))

cv2.waitKey()
