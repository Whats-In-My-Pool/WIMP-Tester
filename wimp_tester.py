import requests
import time
from camera_interface import *


url = 'http://wimpsite.ahines.net/WIMPSite/api/'
#url = 'http://localhost:8000/WIMPSite/api/'

def get_current_tests():
    test_url = url + "test/scheduled_tests/"

    return requests.get(test_url).json()

def get_chemical_tests(id):
    chemical_url = url + "test/test_strip"

    r = requests.get(chemical_url, params={"pk": id})

    return r.json()


def post_result(id, last_run, results):
    post_url = url + "test/report_test/"

    r = requests.post(post_url, json={"scheduled_test_pk": id, "results": results, "time_stamp": last_run})

    print(r.status_code)
    print(r.text)


def run_tests():
    current_strips = get_current_tests()

    for strip in current_strips:
        results = []
        id = strip["pk"]

        image = capture_image()

        for test in get_chemical_tests(strip["fields"]["test_strip"]):
            result = {}
            result["pk"] = test["pk"]

            region = (test["fields"]["region_x1"], test["fields"]["region_y1"], test["fields"]["region_x2"],
                      test["fields"]["region_y2"])

            color = get_result(image, region)

            result["r"] = color[0]
            result["g"] = color[1]
            result["b"] = color[2]

            results.append(result)

        post_result(id, int(time.time()), results)
        print(results)


if __name__ == "__main__":
    run_tests()










