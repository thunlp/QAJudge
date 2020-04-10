import json
import os
import pickle
from sklearn.neural_network import MLPClassifier
from tqdm import trange
import argparse
from usersim import UserSimulator
from ft_tools import get_ft_id, get_ft_num

fold = 1

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-p')
args = parser.parse_args()

user_sim = UserSimulator()

path = args.path

all_data = [[], []]
train_data, train_label = [], []
test_data, test_label = [], []


def eval_function(predict, truth):
    assert len(predict) == len(truth)
    ret = 0.0
    for index in range(get_ft_num()):
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
        # print(res)
        ret += res
    ret /= get_ft_num()
    return ret


def parse(data):
    res = {}
    text = data["ss"]
    res["data"] = []
    for i in range(user_sim.n_questions()):
        res["data"].append(user_sim.answer(text, i) * 2 - 1)
    idx = get_ft_id(data)
    if idx == -1:
        return None
    res["label"] = idx
    return res


def load_data(filename):
    data = []
    label = []
    arr = json.load(open(os.path.join(path, filename), "r", encoding="utf8"))
    for ele in arr:
        result = parse(ele)
        if result == None:
            continue
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
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 256, 256)
    )
    clf.fit(train_data, train_label)
    return clf


if __name__ == "__main__":
    train_data, train_label = load_data("train.json")
    test_data, test_label = load_data("valid.json")

    work(lgb1)
