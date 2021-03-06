import json
import torch
import os
import re

from pytorch_pretrained_bert.tokenization import BertTokenizer

from formatter.Basic import BasicFormatter


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


class BaselineJoint(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        self.tokenizer = BertTokenizer.from_pretrained(config.get("model", "bert_path"))
        self.max_len = config.getint("data", "max_seq_length")
        self.mode = mode
        self.sim = UserSimulator()

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
        ys = []
        ft = []
        zm = []

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
                temp_ys = []
                for a in range(0, self.sim.n_questions()):
                    temp_ys.append(self.sim.answer(text, a))
                ys.append(temp_ys)

                temp_zm = -1
                for label_key in self.label:
                    temp_zm += 1
                    if label_key == temp["meta_info"]["name_of_accusation"]:
                        break
                zm.append(temp_zm)
                ft.append(temp["meta_info"]["law"])

        input_token = torch.LongTensor(input_token)
        if mode != "test":
            ys = torch.LongTensor(ys)
            ft = torch.LongTensor(ft)
            zm = torch.LongTensor(zm)

        if mode != "test":
            return {'input': input, 'token': input_token, "ys": ys, "zm": zm, "ft": ft}
        else:
            return {"input": input, 'token': input_token}
