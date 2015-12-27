from nonogram_detector.binarize_functions import *

from nonogram_detector.nonogram_detector import NonogramDetector
from nonogram_detector.rectangle_functions import *
from nonogram import Nonogram
from perspective_transformer.perspective_transformer import *
from lines_detector.lines_detector import LinesDetector
from lines_detector.line_detection_functions import *

path = "..\\..\\res\\imgs_00\\img_%2d.jpg"
cap = cv2.VideoCapture(path)
while True:
    print
    ret, frame = cap.read()
    if not ret:
        break

    solver = Nonogram(frame)

    nonogram_detector = NonogramDetector(binarize_adaptive_threshold, rectangle_approx_poly)
    solver.set_detector(nonogram_detector)

    transformer = PerspectiveTransformer(binarize_fixed_threshold(96))
    solver.set_perspective_transformer(transformer)

    line_detector = LinesDetector(lines_nxm_kernel)
    solver.set_line_detector(line_detector)

    imgs = solver.solve()
    for img in imgs:
        cv2.imshow('test', img)
        cv2.waitKey(0)
