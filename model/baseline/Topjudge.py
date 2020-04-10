import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.loss import MultiLabelSoftmaxLoss
from model.loss import cross_entropy_loss
from tools.accuracy_init import single_label_top1_accuracy, multi_label_accuracy


class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()

        self.convs = []
        for a in range(2, 6):
            self.convs.append(nn.Conv2d(1, 64, (a, 256)))

        self.convs = nn.ModuleList(self.convs)
        self.feature_len = 256

    def forward(self, x):
        batch_size = x.size()[0]
        x = x.view(batch_size, 1, -1, 256)
        conv_out = []
        gram = 2
        for conv in self.convs:
            y = F.relu(conv(x)).view(batch_size, 64, -1)
            y = F.max_pool1d(y, kernel_size=512 - gram + 1).view(batch_size, -1)
            conv_out.append(y)
            gram += 1

        conv_out = torch.cat(conv_out, dim=1)
        fc_input = conv_out

        features = 256

        fc_input = fc_input.view(-1, features)

        return fc_input


class LSTMDecoder(nn.Module):
    def __init__(self):
        super(LSTMDecoder, self).__init__()
        self.feature_len = 256

        features = 256
        self.hidden_dim = features
        self.outfc = []

        task_name = ["ys", "ft", "zm"]
        num_classes = [54, 23, 20]
        for a in range(0, len(task_name)):
            self.outfc.append(nn.Linear(features, num_classes[a]))

        self.midfc = []
        for x in task_name:
            self.midfc.append(nn.Linear(features, features))

        self.cell_list = [None]
        for x in task_name:
            self.cell_list.append(nn.LSTMCell(256, 256))

        self.hidden_state_fc_list = []
        for a in range(0, len(task_name) + 1):
            arr = []
            for b in range(0, len(task_name) + 1):
                arr.append(nn.Linear(features, features))
            arr = nn.ModuleList(arr)
            self.hidden_state_fc_list.append(arr)

        self.cell_state_fc_list = []
        for a in range(0, len(task_name) + 1):
            arr = []
            for b in range(0, len(task_name) + 1):
                arr.append(nn.Linear(features, features))
            arr = nn.ModuleList(arr)
            self.cell_state_fc_list.append(arr)

        self.outfc = nn.ModuleList(self.outfc)
        self.midfc = nn.ModuleList(self.midfc)
        self.cell_list = nn.ModuleList(self.cell_list)
        self.hidden_state_fc_list = nn.ModuleList(self.hidden_state_fc_list)
        self.cell_state_fc_list = nn.ModuleList(self.cell_state_fc_list)
        self.sigmoid = nn.Sigmoid()

    def init_hidden(self, bs):
        self.hidden_list = []
        task_name = ["ys", "ft", "zm"]
        for a in range(0, len(task_name) + 1):
            self.hidden_list.append((torch.autograd.Variable(torch.zeros(bs, self.hidden_dim).cuda()),
                                     torch.autograd.Variable(torch.zeros(bs, self.hidden_dim).cuda())))

    def forward(self, x):
        fc_input = x
        outputs = []
        task_name = ["ys", "ft", "zm"]
        batch_size = x.size()[0]
        self.init_hidden(batch_size)

        first = []
        for a in range(0, len(task_name) + 1):
            first.append(True)
        for a in range(1, len(task_name) + 1):
            h, c = self.cell_list[a](fc_input, self.hidden_list[a])
            for b in range(1, len(task_name) + 1):
                hp, cp = self.hidden_list[b]
                if first[b]:
                    first[b] = False
                    hp, cp = h, c
                else:
                    hp = hp + self.hidden_state_fc_list[a][b](h)
                    cp = cp + self.cell_state_fc_list[a][b](c)
                self.hidden_list[b] = (hp, cp)
            outputs.append(self.outfc[a - 1](F.relu(self.midfc[a - 1](h))).view(batch_size, -1))

        return outputs


class TopJudge(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(TopJudge, self).__init__()

        self.encoder = CNNEncoder()
        self.decoder = LSTMDecoder()
        self.trans_linear = nn.Linear(self.encoder.feature_len, self.decoder.feature_len)

        self.word_num = 0
        f = open(os.path.join(config.get("model", "bert_path"), "vocab.txt"), "r")
        for line in f:
            self.word_num += 1

        self.embedding = nn.Embedding(self.word_num, 256)

        self.criterion = {
            "ys": MultiLabelSoftmaxLoss(config),
            "ft": cross_entropy_loss,
            "zm": cross_entropy_loss
        }
        self.accuracy_function = {
            "ys": multi_label_accuracy,
            "ft": single_label_top1_accuracy,
            "zm": single_label_top1_accuracy
        }

    def forward(self, data, config, gpu_list, acc_result, mode):
        x = data['token']
        x = self.embedding(x)
        x = self.encoder.forward(x)
        if self.encoder.feature_len != self.decoder.feature_len:
            x = self.trans_linear(x)
        outputs = self.decoder.forward(x)

        loss = 0
        if acc_result is None:
            acc_result = {"ys": None, "ft": None, "zm": None}

        if "ys" in data.keys():
            y = outputs[0]
            y = y.view(y.size()[0], -1, 2)
            label = data["ys"]
            y_out = nn.Softmax(dim=2)(y)
            y_out = y_out[:, :, 1]
            loss += self.criterion["ys"](y, label)
            acc_result["ys"] = self.accuracy_function["ys"](y_out, label, config, acc_result["ys"])

        if "ft" in data.keys():
            y = outputs[1]
            label = data["ft"]
            loss += self.criterion["ft"](y, label)
            acc_result["ft"] = self.accuracy_function["ft"](y, label, config, acc_result["ft"])

        if "zm" in data.keys():
            y = outputs[2]
            label = data["zm"]
            loss += self.criterion["zm"](y, label)
            acc_result["zm"] = self.accuracy_function["zm"](y, label, config, acc_result["zm"])

        return {"loss": loss, "acc_result": acc_result}
