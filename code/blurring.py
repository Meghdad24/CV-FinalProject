import cv2
import numpy as np

from extract import extract
from masking import mask, label_to_points


def blur(image: np.ndarray, points: np.ndarray) -> np.ndarray:

    sigma = 39

    blured_image = cv2.GaussianBlur(image, (sigma, sigma), 0)

    blured_plate = extract(blured_image, points)

    output = mask(image, points, blured_plate)

    return output


# MAIN
if __name__ == '__main__':
    img = cv2.imread("../resource/image/day_12234.jpg")
    cover = cv2.imread("../resource/kntu.jpg")

    with open("../resource/label/126c3652-day_12234.txt", 'r') as file:
        points = label_to_points(file.read().strip().split(" "))

    cv2.imshow("img", blur(img, points))

    cv2.waitKey()
