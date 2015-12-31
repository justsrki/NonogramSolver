import cv2
from config import Config


def binarize_adaptive_threshold(img):
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, Config.block_size, Config.c)


def binarize_fixed_threshold(threshold=128, max_val=255):
    def func(img):
        _, ret = cv2.threshold(img, threshold, max_val, cv2.THRESH_BINARY_INV)
        return ret

    return func
