import json
import torch
import os

from pytorch_pretrained_bert.tokenization import BertTokenizer

from formatter.Basic import BasicFormatter


class BaselineFT(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        self.tokenizer = BertTokenizer.from_pretrained(config.get("model", "bert_path"))
        self.max_len = config.getint("data", "max_seq_length")
        self.mode = mode
        self.label = [
            "非法拘禁",
            "故意伤害",
            "抢劫",
            "抢夺",
            "故意毁坏财物",
            "盗窃",
            "敲诈勒索",
            "诈骗",
            "信用卡诈骗",
            "合同诈骗",
            "掩饰、隐瞒犯罪所得、犯罪所得收益",
            "走私、贩卖、运输、制造毒品",
            "容留他人吸毒",
            "非法持有毒品",
            "危险驾驶",
            "交通肇事",
            "生产、销售假药",
            "非法持有、私藏枪支、弹药",
            "滥伐林木",
            "赌博",
        ]

    def process(self, data, config, mode, *args, **params):
        input = []
        input_token = []
        if mode != "test":
            zm_label = []
            ft_label = []

        for temp in data:
            text = temp[config.get("data", "use_which")]
            input.append(text)

            token = self.tokenizer.tokenize(text)
            token = ["[CLS]"] + token
            while len(token) < self.max_len:
                token.append("[PAD]")
            token = token[0:self.max_len]
            token = self.tokenizer.convert_tokens_to_ids(token)
            input_token.append(token)

            if mode != "test":
                temp_label = -1
                for label_key in self.label:
                    temp_label += 1
                    if label_key == temp["meta_info"]["name_of_accusation"]:
                        break
                zm_label.append(temp_label)
                ft_label.append(temp["meta_info"]["law"])

        input_token = torch.LongTensor(input_token)
        if mode != "test":
            zm_label = torch.LongTensor(zm_label)
            ft_label = torch.LongTensor(ft_label)

        if mode != "test":
            return {'input': input, 'token': input_token, 'zm': zm_label, 'ft': ft_label}
        else:
            return {"input": input, 'token': input_token}
