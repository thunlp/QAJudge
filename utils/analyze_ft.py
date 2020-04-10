import os
import json

pre_path = "" # set your path here
task_list = ["cjo", "cail", "pku"]

possible_set = [
    (196, 0),
    (348, 0),
    (133, 1),
    (264, 0),
    (133, 0),
    (354, 0),
    (234, 0),
    (266, 0),
    (274, 0),
    (347, 0),
    (267, 0),
    (263, 0),
    (238, 0),
    (275, 0),
    (345, 0),
    (312, 0),
    (134, 0),
    (224, 0),
    (141, 0),
    (128, 0),
    (303, 0),
    (150, 0),
    (269, 0),
]

if __name__ == "__main__":
    data = []
    for task in task_list:
        data = data + json.load(open(os.path.join(pre_path, task, "train.json")))
        data = data + json.load(open(os.path.join(pre_path, task, "valid.json")))

    count = {}

    map_list = {}
    more_than = 0
    zero = 0
    for temp in data:
        name = temp["meta_info"]["name_of_accusation"]
        if not (name in map_list.keys()):
            map_list[name] = {}

        se = set()
        for x, y, z in temp["meta_info"]["law"]:
            se.add((x, y))
        temp["meta_info"]["law"] = list(se)

        cnt = 0
        for x, y in temp["meta_info"]["law"]:
            if x < 102 or x > 452:
                continue
            if not ((x, y) in possible_set):
                continue
            cnt += 1
        if cnt == 0:
            zero += 1
            continue
        if cnt > 1:
            more_than += 1
            continue

        for x, y in temp["meta_info"]["law"]:
            if x < 102 or x > 452:
                continue
            if not ((x, y) in map_list[name].keys()):
                map_list[name][(x, y)] = 0
            map_list[name][(x, y)] += 1
            if not ((x, y) in count.keys()):
                count[(x, y)] = 0
            count[(x, y)] += 1

    for name in map_list.keys():
        print(name, len(map_list[name]))
        for a in map_list[name].keys():
            print(a, map_list[name][a])
        print("")

    k = 50
    cnt = 0
    for a in count.keys():
        if count[a] >= k:
            cnt += 1
            print(a, count[a])
    print(cnt)
    print(len(data), more_than, zero)
