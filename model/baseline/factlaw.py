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


class FactLaw(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(FactLaw, self).__init__()

        self.word_num = 0
        f = open(os.path.join(config.get("model", "bert_path"), "vocab.txt"), "r")
        for line in f:
            self.word_num += 1

        self.embedding = nn.Embedding(self.word_num, 256)
        self.encoder = nn.GRU(256, 256, batch_first=True, num_layers=2, bidirectional=True)
        self.law_emb = nn.Embedding(19, 256)
        self.law_token = []
        for a in range(0, 19):
            self.law_token.append(a)
        self.law_token = Variable(torch.LongTensor(self.law_token)).cuda()
        self.attention = Attention()

        self.zm_fc = nn.Linear(1024, 20)

        self.criterion = cross_entropy_loss
        self.accuracy_function = single_label_top1_accuracy

    def init_hidden(self, bs):
        self.hidden = torch.autograd.Variable(torch.zeros(4, bs, 256).cuda())

    def forward(self, data, config, gpu_list, acc_result, mode):
        x = data['token']
        batch_size = x.size()[0]
        x = self.embedding(x)
        self.init_hidden(batch_size)
        h, c = self.encoder(x, self.hidden)
        law_emb = self.law_emb(self.law_token.repeat(batch_size).view(batch_size, -1))

        e = torch.max(h, dim=1)[0]
        a = self.attention(h, law_emb)

        g = torch.bmm(a.permute(0, 2, 1), h)
        g = g.view(batch_size, -1)
        r = torch.mean(h, dim=2)
        er = torch.cat([e, r], dim=1)

        zm = self.zm_fc(er)

        loss = 0

        y = zm
        label = data["zm"]
        loss += self.criterion(y, label)
        acc_result = self.accuracy_function(y, label, config, acc_result)

        return {"loss": loss, "acc_result": acc_result}
