import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.accuracy_init import init_accuracy_function
from model.loss import cross_entropy_loss


class BaselineZMGRU(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(BaselineZMGRU, self).__init__()
        self.emb_dim = 256
        self.hidden_size = 256
        self.output_dim = config.getint("model", "output_dim")
        self.word_num = 0
        f = open(os.path.join(config.get("model", "bert_path"), "vocab.txt"), "r")
        for line in f:
            self.word_num += 1

        self.embedding = nn.Embedding(self.word_num, self.emb_dim)

        self.gru = nn.GRU(self.emb_dim, self.hidden_size, 4, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * self.hidden_size, self.output_dim)

        self.criterion = cross_entropy_loss
        self.accuracy_function = init_accuracy_function(config, *args, **params)

    def forward(self, data, config, gpu_list, acc_result, mode):
        x = data['token']

        x = self.embedding(x)
        y, _ = self.gru(x)
        y = torch.max(y, dim=1)[0]
        y = y.view(y.size()[0], -1)
        y = self.fc(y)

        if config.get("data", "task") in data.keys():
            label = data[config.get("data", "task")]
            loss = self.criterion(y, label)
            acc_result = self.accuracy_function(y, label, config, acc_result)
            return {"loss": loss, "acc_result": acc_result}

        return {}
