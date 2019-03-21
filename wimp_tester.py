import datetime
import random

import requests
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
        tests = get_chemical_tests(strip["fields"]["test_strip"])

        image = capture_image()

        colors = get_results(image)

        if len(colors) != len(tests):
            print("error: incorrect number of tests found")
            return


        for test in get_chemical_tests(strip["fields"]["test_strip"]):
            result = {}
            result["pk"] = test["pk"]

            pos = test["fields"]["test_number"]

            color = colors[len(colors) - pos]

            result["r"] = int(color[0])
            result["g"] = int(color[1])
            result["b"] = int(color[2])

            results.append(result)

        post_result(id, int(time.time()), results)
        print(results)


def dummy_tests():
    current_strips = get_current_tests()

    pk = current_strips[0]["pk"]

    for i in range(1, 6):
        results = []
        for test in get_chemical_tests(current_strips[0]["fields"]["test_strip"]):
            result = {}
            result["pk"] = test["pk"]
            result["r"] = random.randint(0, 255)
            result["g"] = random.randint(0, 255)
            result["b"] = random.randint(0, 255)
            results.append(result)

        post_result(pk, datetime.datetime(2019, i, 14, 12, 0).timestamp(), results)


if __name__ == "__main__":
    run_tests()
