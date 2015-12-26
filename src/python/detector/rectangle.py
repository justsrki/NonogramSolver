import cv2

from util.benchmark import timeit


@timeit
def rectangle_approx_poly(contour):
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx if len(approx) == 4 else None
