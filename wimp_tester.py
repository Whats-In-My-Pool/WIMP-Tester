import requests
import time

url = 'http://wimpsite.ahines.net/WIMPSite/api/'


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


if __name__ == "__main__":
    current_strips = get_current_tests()

    result = {}
    for strip in current_strips:
        results = []
        id = strip["pk"]

        for test in get_chemical_tests(id):
            result = {}
            result["pk"] = test["pk"]
            result["r"] = 128
            result["g"] = 128
            result["b"] = 128

            results.append(result)

        post_result(id, int(time.time()), results)
        print(results)










