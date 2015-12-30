import cv2
import numpy as np
from util.benchmark import timeit


class DigitDetector:
    def __init__(self, classifier):
        self.classifier = classifier
        self.image = None
        self.lines = None

    def set_image(self, image):
        self.image = image

    def set_lines(self, lines, lines_vertical=None, lines_horizontal=None):
        if lines_vertical and lines_horizontal:
            self.lines = (lines_vertical, lines_horizontal)
        else:
            self.lines = lines

    @timeit
    def get_result(self):
        lines_v = self.lines[0]
        lines_h = self.lines[1]

        img_gray = self.image
        self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        # nonogram_values = ([], [])

        for i in xrange(0, len(lines_v) - 1):
            for j in xrange(0, len(lines_h) - 1):
                x1 = lines_v[i] + 1
                x2 = lines_v[i + 1] - 1
                y1 = lines_h[j] + 1
                y2 = lines_h[j + 1] - 1
                p1 = (lines_v[i] + 1, lines_h[j] + 1)
                p2 = (lines_v[i + 1] - 1, lines_h[j + 1] - 1)
                cv2.rectangle(self.image, p1, p2, (255, 0, 0))

                min_h = min(x2-x1, y2-y1) / 3
                min_w = min_h / 2

                region = 255 - img_gray[lines_h[j] + 1:  lines_h[j + 1] - 1, lines_v[i] + 1: lines_v[i + 1] - 1]
                _, contours, _ = cv2.findContours(region.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                # region_values = []
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    if h > min_h and w > min_w:
                        digit = region[y: y + h, x:x + w]
                        value = sorted(self.classifier.classify(255-digit))[0][1]
                        cv2.putText(self.image, str(value), (x+x1, y+y1+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
                        # region_values.append(value)

        return self.image
