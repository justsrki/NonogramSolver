import cv2
import numpy as np
from util.benchmark import timeit
from os import listdir
from os.path import isfile, join
from config import Config


class MaskClassifier:
    @timeit
    def __init__(self, digits_path=None, masks_path=None):
        self.masks = [None] * 10

        if digits_path is not None:
            self.path = digits_path
            self.create_masks()
        elif masks_path is not None:
            # TODO
            pass

    @staticmethod
    def __get_files(path):
        file_paths = []
        for f in listdir(path):
            if isfile(join(path, f)):
                file_paths.append(join(path, f))
            else:
                for digit_path in MaskClassifier.__get_files(join(path, f)):
                    file_paths.append(digit_path)
        return file_paths

    @staticmethod
    def __create_mask(paths):
        mask = np.zeros((Config.digit_height, Config.digit_width), 'float32')
        for path in paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (Config.digit_height, Config.digit_width))
            mask += img

        mask /= mask.max()
        return mask

    def create_masks(self):
        for i in xrange(0, 10):
            paths = MaskClassifier.__get_files(self.path + '\\' + str(i))
            mask = MaskClassifier.__create_mask(paths)
            cv2.imwrite(self.path + '\\masks\\' + str(i) + '.png', mask * 255)
            self.masks[i] = (mask * 255).astype('int16')
        if Config.DEBUG:
            for i in xrange(0, 10):
                print '---------- ', i, ' ---------'
                paths = MaskClassifier.__get_files(self.path + '\\' + str(i))
                self.test(paths, i)

    def test(self, paths, expected):
        for path in paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            digit = min(self.classify(img))
            print "{} - {:.2f}".format(digit[1], digit[0])
            if expected != digit[1]:
                print sorted(self.classify(img))

    def classify(self, img):
        img = cv2.resize(img, (Config.digit_height, Config.digit_width))
        # return [(1.0 * (self.masks[i] * img).sum() / (self.masks[i].sum()), i) for i, mask in enumerate(self.masks)]
        return [(1.0 * np.abs(self.masks[i] - img).sum() / self.masks[i].sum(), i) for i, mask in enumerate(self.masks)]

