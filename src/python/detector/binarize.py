import cv2
from config import Config

from util.benchmark import timeit


@timeit
def binarize_threshold(img):
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, Config.block_size, Config.c)
