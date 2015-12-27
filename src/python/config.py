import cv2


class Config:
    def __init__(self):
        pass

    # resize
    image_width = 2448.0 / 2
    interpolation = cv2.INTER_CUBIC
    cvt_color_code = cv2.COLOR_RGB2GRAY

    # threshold detector
    block_size = 99
    c = 2

    # find rectangles
    d_min = 600
    aspect_ratio = 3
    epsilon = 0.01

    # kernel nxm
    lines_kernel_short = 3
    lines_kernel_long = 512

    # lines
    eps = 12
