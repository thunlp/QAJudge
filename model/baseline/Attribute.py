import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.loss import MultiLabelSoftmaxLoss
from model.loss import cross_entropy_loss
from tools.accuracy_init import single_label_top1_accuracy, multi_label_accuracy


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

        self.w = nn.Linear(512, 256)

    def forward(self, h, u):
        h = self.w(h)
        u = u.permute(0, 2, 1)
        a = torch.bmm(h, u)
        a = torch.softmax(a, dim=1)

        return a


class Attribute(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(Attribute, self).__init__()

        self.word_num = 0
        f = open(os.path.join(config.get("model", "bert_path"), "vocab.txt"), "r")
        for line in f:
            self.word_num += 1

        self.embedding = nn.Embedding(self.word_num, 256)
        self.encoder = nn.LSTM(256, 256, batch_first=True, num_layers=2, bidirectional=True)
        self.ys_emb = nn.Embedding(27, 256)
        self.ys_token = []
        for a in range(0, 27):
            self.ys_token.append(a)
        self.ys_token = Variable(torch.LongTensor(self.ys_token)).cuda()
        self.attention = Attention()

        self.ys_fc = nn.Linear(256 * 27 * 2, 54)
        self.ft_fc = nn.Linear(1024, 23)
        self.zm_fc = nn.Linear(1024, 20)

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

    def init_hidden(self, bs):
        self.hidden = (torch.autograd.Variable(torch.zeros(4, bs, 256).cuda()),
                       torch.autograd.Variable(torch.zeros(4, bs, 256).cuda()))

    def forward(self, data, config, gpu_list, acc_result, mode):
        x = data['token']
        batch_size = x.size()[0]
        x = self.embedding(x)
        self.init_hidden(batch_size)
        h, c = self.encoder(x, self.hidden)
        ys_emb = self.ys_emb(self.ys_token.repeat(batch_size).view(batch_size, -1))

        e = torch.max(h, dim=1)[0]
        a = self.attention(h, ys_emb)

        g = torch.bmm(a.permute(0, 2, 1), h)
        g = g.view(batch_size, -1)
        r = torch.mean(h, dim=2)
        er = torch.cat([e, r], dim=1)

        ys = self.ys_fc(g)
        ft = self.ft_fc(er)
        zm = self.zm_fc(er)

        loss = 0
        if acc_result is None:
            acc_result = {"ys": None, "ft": None, "zm": None}

        if "ys" in data.keys():
            y = ys
            y = y.view(y.size()[0], -1, 2)
            label = data["ys"]
            y_out = nn.Softmax(dim=2)(y)
            y_out = y_out[:, :, 1]
            loss += self.criterion["ys"](y, label)
            acc_result["ys"] = self.accuracy_function["ys"](y_out, label, config, acc_result["ys"])

        if "ft" in data.keys():
            y = ft
            label = data["ft"]
            loss += self.criterion["ft"](y, label)
            acc_result["ft"] = self.accuracy_function["ft"](y, label, config, acc_result["ft"])

        if "zm" in data.keys():
            y = zm
            label = data["zm"]
            loss += self.criterion["zm"](y, label)
            acc_result["zm"] = self.accuracy_function["zm"](y, label, config, acc_result["zm"])

        return {"loss": loss, "acc_result": acc_result}
