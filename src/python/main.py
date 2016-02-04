import os

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

# base path and folder name
base_path = "..\\..\\res\\"
# folder_name, end = "p7_l10", "jpg"
# folder_name, end = "iphone5s", "JPG"
folder_name, end = "i8750", "jpg"

# test and result paths
test_path = base_path + "test\\" + folder_name
result_path = base_path + "result\\" + folder_name + "\\"

# create result folder if doesn't exist
if not os.path.exists(result_path):
    os.mkdir(result_path)


for i, path in enumerate(get_files(test_path, end=end)):
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
        cv2.imwrite(result_path + str(i).zfill(2) + '_' + str(j) + '.png', img)
