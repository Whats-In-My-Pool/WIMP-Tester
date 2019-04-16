import cv2
import numpy as np

radius_avg = 8

def get_contours(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hue, saturation, value = cv2.split(hsv)

    hue = cv2.bitwise_not(hue)

    blur_sat = cv2.GaussianBlur(saturation, (5, 5), 0)

    blur_hue = cv2.GaussianBlur(hue, (5, 5), 0)

    retval, thresholded_sat = cv2.threshold(blur_sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    retval, thresholded_hue = cv2.threshold(blur_hue, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    thresholded = np.add(thresholded_hue, thresholded_sat)

    median_filtered = cv2.medianBlur(thresholded, 5)

    contours, _ = cv2.findContours(median_filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def find_color(contours, image):
    if len(contours) > 0:
        max_contour = contours[0]
        for contour in contours[1:]:
            area = cv2.contourArea(contour)
            if cv2.contourArea(max_contour) < area:
                max_contour = contour

        if cv2.contourArea(max_contour) > 2200:
            y, x, _ = image.shape
            color = get_color_at_point(image, x / 2, y / 2)

        elif cv2.contourArea(max_contour) > 500:
            color = get_color_in_region(image, max_contour)

        else:
            y, x, _ = image.shape
            color = get_color_at_point(image, x/2, y/2)
    else:
        y, x, _ = image.shape

        color = get_color_at_point(image, x/2, y/2)

    return convert_bgr_rgb(color)


def get_color_in_region(image, region):
    # compute the center of the contour

    M = cv2.moments(region)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    avg = get_color_at_point(image, cX, cY)

    return avg


def get_color_at_point(image, x, y):
    x = int(x)
    y = int(y)

    return get_avg(image[y - radius_avg:y + radius_avg, x - radius_avg:x + radius_avg])


def convert_bgr_rgb(color: tuple):
    return color[2], color[1], color[0]


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

    if count != 0:
        return r / count, g / count, b / count
    else:
        return 0, 0, 0


def split_image(image, horz=False):
    width = image.shape[1]
    height = image.shape[0]

    if horz:
        slice_width = int(height/6)
    else:
        slice_width = int(width/6)

    images = []

    for i in range(0, 6):
        if not horz:
            starting_x = slice_width*i
            slice = image[:, starting_x: starting_x + slice_width, :]
        else:
            starting_y = slice_width*i
            slice = image[starting_y: starting_y + slice_width, : :]

        images.append(slice)

    return images


def color_shift_image(color_chart, image, shift=None):
    color_chart = cv2.cvtColor(color_chart, cv2.COLOR_BGR2LAB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    color_chart_mean = get_channels_mean(color_chart)
    color_chart_std = get_channels_std(color_chart)

    image_mean = get_channels_mean(image)
    image_std = get_channels_std(image)

    image_channels = cv2.split(image)

    ndx = 0
    result_channels = []
    for channel in image_channels:
        if shift:
            channel = np.add(channel, shift[ndx])

        result_channel = np.subtract(channel, image_mean[ndx])

        result_channel = np.multiply(result_channel, image_std[ndx] / (color_chart_std[ndx] * 0.15))
        result_channel = np.add(result_channel, color_chart_mean[ndx])
        result_channel = np.clip(result_channel, 0, 255)

        result_channels.append(result_channel)
        ndx += 1

    result_channels[0] = result_channels[0]
    result = cv2.merge(result_channels)

    result = cv2.cvtColor(result.astype("uint8"), cv2.COLOR_LAB2BGR)
    return result


def get_channels_mean(image):
    avg = []
    for channel in cv2.split(image):
        avg.append(np.mean(channel))

    return avg


def get_channels_std(image):
    std = []
    for channel in cv2.split(image):
        std.append(np.std(channel))

    return std