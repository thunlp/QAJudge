import logging

from .Basic import BasicFormatter
from formatter.ys_predict.YSBert import YSBert
from formatter.xq_predict.XQBert import XQBert
from formatter.zm_predict.ZMBert import ZMBert
from formatter.zm_predict.ZMDqn import ZMDqn
from formatter.ft_predict.FTDqn import FTDqn
from formatter.baseline.ZM import BaselineZM
from formatter.baseline.YS import BaselineYS
from formatter.baseline.FT import BaselineFT
from formatter.baseline.Joint import BaselineJoint
from formatter.baseline.BertQA import BaselineBertQA

logger = logging.getLogger(__name__)

formatter_list = {
    "Basic": BasicFormatter,
    "YSBert": YSBert,
    "XQBert": XQBert,
    "ZMBert": ZMBert,
    "ZMDqn": ZMDqn,
    "FTDqn": FTDqn,
    "BaselineZM": BaselineZM,
    "BaselineYS": BaselineYS,
    "BaselineFT": BaselineFT,
    "BaselineJoint": BaselineJoint,
    "BaselineBertQA": BaselineBertQA
}


def init_formatter(config, mode, *args, **params):
    temp_mode = mode
    if mode != "train":
        try:
            config.get("data", "%s_formatter_type" % temp_mode)
        except Exception as e:
            logger.warning(
                "[reader] %s_formatter_type has not been defined in config file, use [dataset] train_formatter_type instead." % temp_mode)
            temp_mode = "train"
    which = config.get("data", "%s_formatter_type" % temp_mode)

    if which in formatter_list:
        formatter = formatter_list[which](config, mode, *args, **params)

        return formatter
    else:
        logger.error("There is no formatter called %s, check your config." % which)
        raise NotImplementedError
