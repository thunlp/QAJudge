import re
import os
import json


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


pre_path = "" # set your path here
task_list = ["cjo", "cail", "pku"]

if __name__ == "__main__":
    for task in task_list:
        data = []
        data = data + json.load(open(os.path.join(pre_path, task, "train.json")))
        data = data + json.load(open(os.path.join(pre_path, task, "valid.json")))

        sim = UserSimulator()

        max_cnt = 0
        total_cnt = 0
        arr = []
        for a in range(0, len(data)):
            cnt = 0
            for b in range(0, 27):
                cnt += sim.answer(data[a]["ss"], b)

            max_cnt = max(max_cnt, cnt)
            total_cnt += cnt
            arr.append(cnt)

        arr.sort()
        print(task, len(data), max_cnt, total_cnt / len(data))
        cx = []
        for a in range(0, 10):
            cx.append(arr[int(a / 10 * len(arr))])
        print(cx)
        cx = []
        for a in range(0, 28):
            cx.append(0)
        for a in range(0, len(arr)):
            cx[arr[a]] = (a + 1) / len(arr)
        p = 27
        while p > 0:
            if cx[p] == 0:
                cx[p] = cx[p - 1]
        print(cx)
