import os
import json

pre_path = "" # set your path here
task_list = ["cjo", "cail", "pku"]

if __name__ == "__main__":
    for task in task_list:
        data = []
        data = data + json.load(open(os.path.join(pre_path, task, "train.json")))
        data = data + json.load(open(os.path.join(pre_path, task, "valid.json")))

        max_len = 0
        total_len = 0
        for a in range(0, len(data)):
            l = len(data[a]["ss"])
            max_len = max(max_len, l)
            total_len += l

        print(task, len(data), max_len, total_len / len(data))
