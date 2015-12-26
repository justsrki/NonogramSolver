from detector.binarize import *

from detector.detector import Detector
from detector.rectangle import *
from nonogram import Nonogram
from perspective_transformer.perspective_transformer import *

path = "..\\..\\res\\imgs_00\\img_%2d.jpg"
cap = cv2.VideoCapture(path)
while True:
    print
    ret, frame = cap.read()
    if not ret:
        break

    solver = Nonogram(frame)
    solver.set_detector(Detector(binarize_threshold, rectangle_approx_poly))

    transformer = PerspectiveTransformer()
    solver.set_perspective_transformer(transformer)

    img = solver.solve()
    cv2.imshow('test', img)
    # img, contours = solver.solve()
    # cv2.imshow('test', cv2.drawContours(cv2.cvtColor(solver.detector.image, cv2.COLOR_GRAY2RGB), contours, -1, (255, 0, 0), thickness=3))
    cv2.waitKey(0)
