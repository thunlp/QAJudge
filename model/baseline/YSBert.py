import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel

from tools.accuracy_init import init_accuracy_function
from model.loss import MultiLabelSoftmaxLoss


class BaselineYSBert(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(BaselineYSBert, self).__init__()

        self.output_dim = config.getint("model", "output_dim")
        self.bert = BertModel.from_pretrained(config.get("model", "bert_path"))
        self.fc = nn.Linear(768, self.output_dim * 2)

        self.criterion = MultiLabelSoftmaxLoss(config)
        self.accuracy_function = init_accuracy_function(config, *args, **params)

    def init_multi_gpu(self, device, config, *args, **params):
        self.bert = nn.DataParallel(self.bert, device_ids=device)

    def forward(self, data, config, gpu_list, acc_result, mode):
        x = data['token']

        _, y = self.bert(x, output_all_encoded_layers=False)
        y = y.view(y.size()[0], -1)
        y = self.fc(y)
        y = y.view(y.size()[0], -1, 2)

        if "label" in data.keys():
            label = data["label"]
            y_out = nn.Softmax(dim=2)(y)
            y_out = y_out[:, :, 1]
            loss = self.criterion(y, label)
            acc_result = self.accuracy_function(y_out, label, config, acc_result)
            return {"loss": loss, "acc_result": acc_result}

        return {}
