import numpy as np
import cv2
import time

LED_PIN = 18

alpha = 1
beta = 2.0

radius_avg = 20
crop_box = 0

def find_regions(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hue, saturation, value = cv2.split(hsv)

    hue = cv2.bitwise_not(hue)

    blur_sat = cv2.GaussianBlur(saturation, (5, 5), 0)

    blur_hue = cv2.GaussianBlur(hue, (5, 5), 0)

    retval, thresholded_sat = cv2.threshold(blur_sat, 70, 255, cv2.THRESH_BINARY)

    retval, thresholded_hue = cv2.threshold(blur_hue, 200, 255, cv2.THRESH_BINARY)

    thresholded = np.add(thresholded_hue, thresholded_sat)

    median_filtered = cv2.medianBlur(thresholded, 5)

    contours, hierarchy = cv2.findContours(median_filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_list = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            contour_list.append(contour)

    return contour_list


def get_color_in_regions(image, regions):
    results = []

    for c in regions:
        # compute the center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        avg = get_avg(image[cY - radius_avg:cY + radius_avg, cX - radius_avg:cX + radius_avg])

        results.append(convert_bgr_rbg(avg))

    return results


def get_results(image):

    regions = find_regions(image)

    colors = get_color_in_regions(image, regions)

    return colors


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

        stream = io.BytesIO()

        with PiCamera() as camera:
            time.sleep(3)
            camera.resolution = (820, 616)

            with PiRGBArray(camera) as stream:
                camera.capture(stream, format='bgr')
                time.sleep(1)
                image = stream.array

        GPIO.output(18, 0)

        time.sleep(2)

        GPIO.cleanup()

        rows, cols, _ = image.shape

        rotM = cv2.getRotationMatrix2D((cols / 2, rows / 2), 4, 1)
        rot_image = cv2.warpAffine(image, rotM, (cols, rows))

        rot_image = rot_image[187:255, 162:786]

        cv2.imwrite("tmp.png", rot_image)
        return image
    except Exception as e:
        print(e)
        return cv2.imread("test_rig.png")


def demo(img):
    # Read image
    image = cv2.imread(img)


    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imshow('HSV Image', hsv)
    cv2.waitKey(0)

    hue, saturation, value = cv2.split(hsv)

    # blur_val = cv2.GaussianBlur(value, (5, 5), 0)

    #saturation = cv2.bitwise_not(saturation)

    hue = cv2.bitwise_not(hue)
    value = cv2.bitwise_not(value)

    cv2.imshow('Value Image', value)

    cv2.imshow('Hue Image', hue)
    cv2.imshow('Saturation', saturation)
    cv2.waitKey(0)

    blur_sat = cv2.GaussianBlur(saturation, (5, 5), 0)

    blur_hue = cv2.GaussianBlur(hue, (5, 5), 0)

    #blur_val = cv2.GaussianBlur(value, (5, 5), 0)

    #retval, thresholded_val = cv2.threshold(blur_val, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #cv2.imshow('Thresholded Image', thresholded_val)
    #cv2.waitKey(0)

    retval, thresholded_sat = cv2.threshold(blur_sat, 70, 255, cv2.THRESH_BINARY )
    cv2.imshow('Thresholded Sat Image', thresholded_sat)
    cv2.waitKey(0)

    retval, thresholded_hue = cv2.threshold(blur_hue, 200, 255, cv2.THRESH_BINARY)
    cv2.imshow('Thresholded Hue Image', thresholded_hue)
    cv2.waitKey(0)

    thresholded = np.add(thresholded_hue, thresholded_sat)
    #thresholded = thresholded_sat

    medianFiltered = cv2.medianBlur(thresholded, 5)
    cv2.imshow('Median Filtered Image', medianFiltered)
    cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(medianFiltered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_list = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            contour_list.append(contour)

    cv2.drawContours(image, contour_list, -1, (255, 0, 0), 2)
    cv2.imshow('Objects Detected', image)
    cv2.waitKey(0)
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
    demo("tmp.png")
    # image = capture_image()
    # get_results(image)
