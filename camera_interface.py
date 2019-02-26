import numpy as np
from PIL import Image


def get_result(image, region):
    test = image.crop(region)

    test.save("{}.jpg".format(region))

    return get_adjusted_avg(np.array(test))


def get_avg(array):
    r = 0
    g = 0
    b = 0
    count = 0

    for arr in array:
        for rgb in arr:
            r += rgb[0]
            g += rgb[1]
            b += rgb[2]

            count += 1

    return r / count, g / count, b / count


def get_adjusted_avg(array):
    r = 0
    g = 0
    b = 0
    count = 0

    avg_r, avg_g, avg_b = get_avg(array)

    for arr in array:
        for rgb in arr:
            if get_error(rgb[0], avg_r) < 0.10 and get_error(rgb[1], avg_b) < 0.10 and get_error(rgb[2], avg_g) < 0.10:
                r += rgb[0]
                g += rgb[1]
                b += rgb[2]

                count += 1

    if count != 0:
        return r/count, g/count, b/count
    else:
        return avg_r, avg_g, avg_b


def get_error(a, b):
    return (a - b)/b * 100

def capture_image():
    return Image.open("test.jpg")


