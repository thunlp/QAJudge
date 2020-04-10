import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel

from tools.accuracy_init import init_accuracy_function
from model.loss import MultiLabelSoftmaxLoss
from model.loss import cross_entropy_loss


class BaselineZMBert(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(BaselineZMBert, self).__init__()

        self.output_dim = config.getint("model", "output_dim")
        self.bert = BertModel.from_pretrained(config.get("model", "bert_path"))
        self.fc = nn.Linear(768, self.output_dim)

        self.criterion = cross_entropy_loss
        self.accuracy_function = init_accuracy_function(config, *args, **params)

    def init_multi_gpu(self, device, config, *args, **params):
        self.bert = nn.DataParallel(self.bert, device_ids=device)

    def forward(self, data, config, gpu_list, acc_result, mode):
        x = data['token']

        _, y = self.bert(x, output_all_encoded_layers=False)
        y = y.view(y.size()[0], -1)
        y = self.fc(y)
        y = y.view(y.size()[0], -1)

        if config.get("data", "task")in data.keys():
            label = data[config.get("data", "task")]
            loss = self.criterion(y, label)
            acc_result = self.accuracy_function(y, label, config, acc_result)
            return {"loss": loss, "acc_result": acc_result}

        return {}
