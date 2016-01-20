from nonogram import Nonogram
from nonogram_detector.binarize_functions import *
from nonogram_detector.nonogram_detector import *
from nonogram_detector.rectangle_functions import *
from perspective_transformer.perspective_transformer import *
from lines_detector.lines_detector import *
from lines_detector.line_detection_functions import *
from digit_detector.digit_detector import *
from digit_classifier.mask_classifier import *
from nonogram_solver.nonogram_solver import *
from solution_creator.solution_creator import *
from util.file_reader import *


# imgs_path, end = "..\\..\\res\\test_iphone5s", "JPG"
imgs_path, end = "..\\..\\res\\test_i8750", "jpg"
# imgs_path, end = "..\\..\\res\\imgs_00", "jpg"

for i, path in enumerate(get_files(imgs_path, end=end)):

    nonogram_detector = NonogramDetector(binarize_adaptive_threshold, rectangle_approx_poly)
    transformer = PerspectiveTransformer(binarize_fixed_threshold(200), binarize_adaptive_threshold)
    line_detector = LinesDetector(lines_nxm_kernel)
    classifier = MaskClassifier(masks_path="..\\..\\res\\digits_00\\masks")
    digit_detector = DigitDetector(classifier)
    nonogram_solver = NonogramSolver()
    solution_creator = SolutionCreator()

    solver = Nonogram(path, nonogram_detector=nonogram_detector, perspective_transformer=transformer,
                      line_detector=line_detector, digit_classifier=classifier, digit_detector=digit_detector,
                      nonogram_solver=nonogram_solver, solution_creator=solution_creator)

    imgs = solver.solve()
    for j, img in enumerate(imgs):
        cv2.imwrite(str(i).zfill(2) + '_' + str(j) + '.png', img)
