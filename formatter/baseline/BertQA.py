import json
import torch
import os
import re

from pytorch_pretrained_bert.tokenization import BertTokenizer

from formatter.Basic import BasicFormatter

question_list = [
    "是否非法占有",
    "是否趁其不备非法占有",
    "是否涉及财务",
    "是否损害他人身体健康",
    "是否有暴力行为",
    "是否存在胁迫情节",
    "是否主观故意",
    "是否涉及毒品",
    "是否涉及交易",
    "是否有容留情节",
    "是否和交通相关",
    "是否为酒后行为",
    "是否造成事故",
    "是否虚构事实",
    "是否涉及假药",
    "是否涉及信用卡",
    "是否涉及合同",
    "是否涉枪支弹药",
    "是否有砍伐情节",
    "是否无证砍伐",
    "是否有具体金额",
    "是否涉赌",
    "是否限制他人人身自由",
    "是否有恐吓、威胁、要挟情节",
    "是否有隐瞒情节",
    "是否存在犯罪所得",
    "是否毁坏物品",
]


class UserSimulator:
    def __init__(self):
        self.pat = [  # n_action=27
            r'(.*)(偷|盗|窃|抢|劫|诈骗|骗取|非法占有)(.*)',  # 非法占有
            r'(.*)(偷|盗|窃)(.*)',  # 偷窃（趁其不备地非法占有）
            r'(.*)(财物|金额)(.*)',  # 涉及财物
            r'(.*)(伤)(.*)',  # 损害他人身体健康
            r'(.*)(杀|伤害|抢劫|砍|强奸|刺|殴打)(.*)',  # 暴力
            r'(.*)(胁迫|威胁|强迫|逼迫)(.*)',  # 胁迫
            r'(.*)(故意)(.*)',  # 故意
            r'(.*)(毒品)(.*)',  # 涉及毒品
            r'(.*)(卖|出售|交易)(.*)',  # 涉及交易
            r'(.*)(容留)(.*)',  # 容留
            r'(.*)(交通|驾驶)(.*)',  # 交通
            r'(.*)(乙醇|酒后|醉酒)(.*)',  # 酒后
            r'(.*)(事故)(.*)',  # 事故
            r'(.*)(骗|诈|谎|虚构)(.*)',  # 虚构事实
            r'(.*)(假药)(.*)',  # 是否涉及假药
            r'(.*)(信用卡)(.*)',  # 是否涉及信用卡
            r'(.*)(合同)(.*)',  # 是否涉及合同
            r'(.*)(枪支|弹药)(.*)',  # 是否涉枪支、弹药
            r'(.*)((砍|采|滥)伐)(.*)',  # 是否砍伐
            r'(.*)((未|没|无)([^。，]*)林木采伐许可证)(.*)',  # 是否无证砍伐
            r'(.*)(\d元)(.*)',  # 是否有具体金额
            r'(.*)(赌)(.*)',  # 是否涉赌
            r'(.*)(((限制|剥夺)([^。，]*)人身自由)|拘禁)(.*)',  # 是否限制他人人身自由
            r'(.*)(恐吓|威胁|要挟|敲诈|勒索)(.*)',  # 是否有恐吓、威胁、要挟情节
            r'(.*)(明知|掩饰|瞒|假装)(.*)',  # 隐瞒
            r'(.*)(犯罪所得|赃物)(.*)',  # 是否存在犯罪所得
            r'(.*)(砸(碎|损|烂)|毁|损坏)(.*)',  # 是否毁坏物品
        ]

    def answer(self, text, question):
        return int(bool(re.match(self.pat[question], text)))

    def n_questions(self):
        return 27


class BaselineBertQA(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        self.tokenizer = BertTokenizer.from_pretrained(config.get("model", "bert_path"))
        self.max_len = config.getint("data", "max_seq_length")
        self.mode = mode
        self.sim = UserSimulator()

    def process(self, data, config, mode, *args, **params):
        input = []
        input_token = []
        token_type = []

        if mode != "test":
            label = []

        for temp in data:
            text = temp[config.get("data", "use_which")]
            input.append(text)

            temp_input_token = []
            temp_token_type = []
            for question_id in range(0, self.sim.n_questions()):
                text_token = self.tokenizer.tokenize(text)
                question_token = self.tokenizer.tokenize(question_list[question_id])
                text_token = text_token[0:self.max_len - len(question_token)]

                ix, tx = [], []
                for a in range(0, len(text_token)):
                    ix.append(text_token[a])
                    tx.append(0)
                for a in range(0, len(question_token)):
                    ix.append(question_token[a])
                    tx.append(1)
                while len(ix) < self.max_len:
                    ix.append("[PAD]")
                    tx.append(1)

                ix = self.tokenizer.convert_tokens_to_ids(ix)
                temp_input_token.append(ix)
                temp_token_type.append(tx)

            input_token.append(temp_input_token)
            token_type.append(temp_token_type)

            if mode != "test":
                temp_label = []
                for a in range(0, self.sim.n_questions()):
                    temp_label.append(self.sim.answer(text, a))
                label.append(temp_label)

        input_token = torch.LongTensor(input_token)
        token_type = torch.LongTensor(token_type)
        if mode != "test":
            label = torch.LongTensor(label)

        if mode != "test":
            return {'input': input, 'token': input_token, 'type': token_type, 'label': label}
        else:
            return {"input": input, 'token': input_token, 'type': token_type}
