import json

from .accuracy_tool import gen_micro_macro_result


def null_output_function(data, config, *args, **params):
    return ""


def basic_output_function(data, config, *args, **params):
    which = config.get("output", "output_value").replace(" ", "").split(",")
    temp = gen_micro_macro_result(data)
    result = {}
    for name in which:
        result[name] = temp[name]

    return json.dumps(result, sort_keys=True)


def joint_output_function(data, config, *args, **params):
    result = {}
    which = "micro_f1,macro_precision,macro_recall,macro_f1".split(",")
    for task in ["ys", "ft", "zm"]:
        result[task] = []
        temp = gen_micro_macro_result(data[task])
        for name in which:
            result[task].append(round(temp[name], 4))

    return json.dumps(result, sort_keys=True)
