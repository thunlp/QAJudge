import json
import os

prefix = "" # set your path here

file_list = ["data.json", "train.json", "valid.json"]

for filename in file_list:
    data = json.load(open(os.path.join(prefix, filename), "r"))
    for x in data:
        arr = x["content"].split("h2>")
        x.pop("content")
        x["ss"] = arr[2].replace("\n", "").replace("<p>", "").replace("</p>", "").replace("<","")
        x["ly"] = arr[4].replace("\n", "").replace("<p>", "").replace("</p>", "").replace("<","")

    json.dump(data, open(os.path.join(prefix, filename), "w"), indent=2, ensure_ascii=False, sort_keys=True)
