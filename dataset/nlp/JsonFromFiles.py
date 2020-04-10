import json
import os
from torch.utils.data import Dataset

from tools.dataset_tool import dfs_search
from tools.ft_tools import get_ft_id


class JsonFromFilesDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.file_list = []
        self.data_path = config.get("data", "%s_data_path" % mode)
        self.encoding = encoding

        filename_list = config.get("data", "%s_file_list" % mode).replace(" ", "").split(",")
        recursive = config.getboolean("data", "recursive")

        for name in filename_list:
            self.file_list = self.file_list + dfs_search(os.path.join(self.data_path, name), recursive)
        self.file_list.sort()

        self.load_mem = config.getboolean("data", "load_into_mem")
        self.json_format = config.get("data", "json_format")

        self.data = []
        for filename in self.file_list:
            if self.json_format == "single":
                self.data = self.data + json.load(open(filename, "r", encoding=encoding))
            else:
                f = open(filename, "r", encoding=encoding)
                for line in f:
                    self.data.append(json.loads(line))

        self.filter_data()
        print(len(self.data))

    def filter_data(self):
        if self.config.get("data", "task").find("ft") == -1:
            return
        data = []
        for temp in self.data:
            idx = get_ft_id(temp)
            if idx == -1:
                continue
            temp["meta_info"]["law"] = idx
            data.append(temp)
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
