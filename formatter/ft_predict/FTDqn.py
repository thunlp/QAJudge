import json
import torch
import os

from pytorch_pretrained_bert.tokenization import BertTokenizer

from formatter.Basic import BasicFormatter


class FTDqn(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        self.tokenizer = BertTokenizer.from_pretrained(config.get("model", "bert_path"))
        self.max_len = config.getint("data", "max_seq_length")
        self.mode = mode

    def process(self, data, config, mode, *args, **params):
        input = []
        input_token = []
        if mode != "test":
            label = []

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
                temp_label = temp["meta_info"]["law"]
                assert temp_label!=-1
                label.append(temp_label)

        input_token = torch.LongTensor(input_token)
        if mode != "test":
            label = torch.LongTensor(label)

        if mode != "test":
            return {'input': input, 'token': input_token, 'label': label}
        else:
            return {"input": input, 'token': input_token}
