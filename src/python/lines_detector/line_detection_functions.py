import cv2
import numpy as np
from config import Config


def _applay_nxm_kernel(image, x, y):
    image = cv2.Sobel(image, -1, 0, 2) if x > y else cv2.Sobel(image, -1, 2, 0)
    image = cv2.blur(image, (x, y))
    _, image = cv2.threshold(image, 255 / min(x, y) * 0.75, 255, cv2.THRESH_BINARY)
    # TODO: move to config
    v = 7
    return cv2.erode(cv2.dilate(image, np.ones((v, v), 'uint8')), np.ones((v, v), 'uint8'))


def lines_nxm_kernel(image):
    img_vertical = _applay_nxm_kernel(image, Config.lines_kernel_short, Config.lines_kernel_long)
    img_horizontal = _applay_nxm_kernel(image, Config.lines_kernel_long, Config.lines_kernel_short)
    return img_vertical, img_horizontal
