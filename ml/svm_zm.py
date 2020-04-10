import json
import os
import pickle
from sklearn.svm import SVC
from tqdm import trange
import argparse
from usersim import UserSimulator

fold = 1

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-p')
args = parser.parse_args()

user_sim = UserSimulator()

path = args.path

all_data = [[], []]
train_data, train_label = [], []
test_data, test_label = [], []

all_labels = [
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

label2id = {all_labels[i]: i for i in range(len(all_labels))}


def eval_function(predict, truth):
    assert len(predict) == len(truth)
    ret = 0.0
    for index in range(len(all_labels)):
        Aa = 0
        Na = 0
        An = 0
        Nn = 0
        for a in range(0, len(predict)):
            if int(predict[a]) == index:
                if int(truth[a]) == index:
                    Aa += 1
                else:
                    An += 1
            else:
                if int(truth[a]) == index:
                    Na += 1
                else:
                    Nn += 1
        a = Aa + Na
        n = An + Nn
        A = Aa + An
        N = Na + Nn

        res = 2.0 * Aa / (A + a)
        ret += res
    ret /= len(all_labels)
    return ret


def parse(data):
    res = {}
    text = data["ss"]
    res["data"] = []
    for i in range(user_sim.n_questions()):
        res["data"].append(user_sim.answer(text, i) * 2 - 1)
    res["label"] = label2id[data["meta_info"]["name_of_accusation"]]
    return res


def load_data(filename):
    data = []
    label = []
    arr = json.load(open(os.path.join(path, filename), "r", encoding="utf8"))
    for ele in arr:
        result = parse(ele)
        data.append(result["data"])
        label.append(result["label"])
    return data, label


def work(method_function):
    model = method_function(train_data, train_label, test_data, test_label)
    predict = model.predict(test_data)
    f1 = eval_function(predict, test_label)
    print(f1)
    return model, f1


def lgb1(train_data, train_label, test_data, test_label):
    clf = SVC()
    clf.fit(train_data, train_label)
    return clf


if __name__ == "__main__":
    train_data, train_label = load_data("train.json")
    test_data, test_label = load_data("valid.json")

    work(lgb1)
