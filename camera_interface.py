import numpy as np
import cv2
import time

LED_PIN = 18

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
        img = cv2.imread("new_tmp.png")
        return img

