from model.ys_predict.Bert import YSBert
from model.xq_predict.Bert import XQBert
from model.zm_predict.Bert import ZMBert
from model.zm_predict.dqn import ZMDqn
from model.baseline.ZMBert import BaselineZMBert
from model.baseline.YSBert import BaselineYSBert
from model.baseline.ZMDPCNN import BaselineZMDPCNN
from model.baseline.YSDPCNN import BaselineYSDPCNN
from model.baseline.ZMGRU import BaselineZMGRU
from model.baseline.YSGRU import BaselineYSGRU
from model.baseline.Topjudge import TopJudge
from model.baseline.Attribute import Attribute
from model.baseline.BertQA import BaselineBertQA
from model.baseline.factlaw import FactLaw

model_list = {
    "YSBert": YSBert,
    "XQBert": XQBert,
    "ZMBert": ZMBert,
    "ZMDqn": ZMDqn,
    "BaselineZMBert": BaselineZMBert,
    "BaselineYSBert": BaselineYSBert,
    "BaselineZMDPCNN": BaselineZMDPCNN,
    "BaselineYSDPCNN": BaselineYSDPCNN,
    "BaselineZMGRU": BaselineZMGRU,
    "BaselineYSGRU": BaselineYSGRU,
    "TopJudge": TopJudge,
    "Attribute": Attribute,
    "BertQA": BaselineBertQA,
    "FactLaw": FactLaw
}


def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
