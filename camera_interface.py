import numpy as np
import cv2
import time
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

LED_PIN = 18

alpha = 1
beta = 2.0

radius_avg = 3
crop_box = 0


def get_delta_e(color1, color2):
    color1_rgb = sRGBColor(color1[0] / 255.0, color1[1] / 255.0, color1[2] / 255.0)
    color2_rgb = sRGBColor(color2[0] / 255.0, color2[1] / 255.0, color2[2] / 255.0)

    color1_lab = convert_color(color1_rgb, LabColor)

    color2_lab = convert_color(color2_rgb, LabColor)

    return delta_e_cie2000(color1_lab, color2_lab)


def show_image(image):
    cv2.imshow("Output", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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

def find_color(image):
    contours = get_contours(image)

    if len(contours) > 0:
        max_contour = contours[0]
        for contour in contours[1:]:
            area = cv2.contourArea(contour)
            if cv2.contourArea(max_contour) < area:
                max_contour = contour

        if cv2.contourArea(max_contour) > 2000:
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

    return convert_bgr_rbg(color)


def get_color_at_point(image, x, y):
    x = int(x)
    y = int(y)

    return get_avg(image[y - radius_avg:y + radius_avg, x - radius_avg:x + radius_avg])


def get_color_in_region(image, region):
    # compute the center of the contour

    M = cv2.moments(region)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    avg = get_color_at_point(image, cX, cY)

    return avg


def split_image(image):
    width = image.shape[1]
    slice_width = int(width/6)

    images = []

    for i in range(0, 6):
        starting_x = slice_width*i
        slice = image[:, starting_x: starting_x + slice_width, :]
        images.append(slice)

    return images


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
        return r / count, g / count, b / count
    else:
        return avg_r, avg_g, avg_b


def get_error(a, b):
    return (a - b) / b * 100


def convert_bgr_rbg(bgr):
    return bgr[2], bgr[1], bgr[0]


def color_shift_image(color_chart, image):
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
        result_channel = np.subtract(channel, image_mean[ndx])
        
        result_channel = np.multiply(result_channel, image_std[ndx]/(color_chart_std[ndx]*0.15))
        result_channel = np.add(result_channel, color_chart_mean[ndx])
        result_channel = np.clip(result_channel, 0, 255)

        result_channels.append(result_channel)
        ndx += 1


    result_channels[0] = result_channels[0] * 0.75
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


def capture_image():
    try:
        from picamera.array import PiRGBArray
        from picamera import PiCamera
        import RPi.GPIO as GPIO
        import io

        GPIO.setmode(GPIO.BCM)

        GPIO.setup(18, GPIO.OUT, initial=GPIO.LOW)
        GPIO.output(18, 1)
        time.sleep(1)

        with PiCamera() as camera:
            camera.resolution = (832, 624)
            camera.shutter_speed = 500
            image = np.empty((624, 832, 3), dtype=np.uint8)

            camera.shutter_speed = 500

            time.sleep(3)

            camera.capture(image, format='bgr')
            time.sleep(1)

        GPIO.output(18, 0)

        time.sleep(2)

        GPIO.cleanup()

        rows, cols, _ = image.shape

        rotM = cv2.getRotationMatrix2D((cols / 2, rows / 2), -92, 1)
        rot_image = cv2.warpAffine(image, rotM, (cols, rows))

        rot_image = rot_image[310:375, 102:680]

        cv2.imwrite("tmp.png", rot_image)
        return image
    except Exception as e:
        print(e)
        img = cv2.imread("tmp.png")
        return img


def demo(img):
    # Read image
    image = cv2.imread(img)


    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imshow('HSV Image', hsv)

    hue, saturation, value = cv2.split(hsv)

    # blur_val = cv2.GaussianBlur(value, (5, 5), 0)

    #saturation = cv2.bitwise_not(saturation)

    hue = cv2.bitwise_not(hue)
    value = cv2.bitwise_not(value)

    cv2.imshow('Value Image', value)

    cv2.imshow('Hue Image', hue)
    cv2.imshow('Saturation', saturation)

    blur_sat = cv2.GaussianBlur(saturation, (5, 5), 0)

    blur_hue = cv2.GaussianBlur(hue, (5, 5), 0)

    #blur_val = cv2.GaussianBlur(value, (5, 5), 0)

    #retval, thresholded_val = cv2.threshold(blur_val, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #cv2.imshow('Thresholded Image', thresholded_val)
    #cv2.waitKey(0)

    retval, thresholded_sat = cv2.threshold(blur_sat, 70, 255, cv2.THRESH_BINARY)
    cv2.imshow('Thresholded Sat Image', thresholded_sat)

    retval, thresholded_hue = cv2.threshold(blur_hue, 200, 255, cv2.THRESH_BINARY)
    cv2.imshow('Thresholded Hue Image', thresholded_hue)

    thresholded = np.add(thresholded_hue, thresholded_sat)
    #thresholded = thresholded_sat

    medianFiltered = cv2.medianBlur(thresholded, 5)
    cv2.imshow('Median Filtered Image', medianFiltered)

    contours, hierarchy = cv2.findContours(medianFiltered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_list = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            contour_list.append(contour)

    cv2.drawContours(image, contour_list, -1, (255, 0, 0), 2)
    cv2.imshow('Objects Detected', image)
    for c in contour_list:
        # compute the center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        avg = get_avg(image[cY - radius_avg:cY + radius_avg, cX - radius_avg:cX + radius_avg])

        avg = (int(avg[2]), int(avg[1]), int(avg[0]))

        # draw the contour and center of the shape on the image
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
        cv2.putText(image, str(avg), (cX - 20, cY - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


    # show the image
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    img = capture_image()

    #img = cv2.imread("borb.jpg")
    #color_chart = cv2.imread("sunset.jpg")
    color_chart = cv2.imread("color_chart.png")

    result = color_shift_image(color_chart, img)

    cv2.imshow("Output", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("shifted_output.png", result)

    splits = split_image(result)

