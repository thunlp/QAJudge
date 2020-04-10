possible_set = [
    (196, 0),
    (348, 0),
    # (133, 1),
    (264, 0),
    (133, 0),
    (354, 0),
    (234, 0),
    (266, 0),
    (274, 0),
    (347, 0),
    (267, 0),
    (263, 0),
    (238, 0),
    (275, 0),
    (345, 0),
    (312, 0),
    # (134, 0),
    (224, 0),
    (141, 0),
    (128, 0),
    (303, 0),
    # (150, 0),
    # (269, 0),
]


def get_ft_id(data):
    cnt = 0
    ft_list = data["meta_info"]["law"]
    se = set()
    for x, y, z in ft_list:
        se.add((x, y))
    temp = list(se)
    for x, y in temp:
        if (x, y) in possible_set:
            cnt += 1
    if cnt != 1:
        return -1

    for x, y in temp:
        if (x, y) in possible_set:
            for a in range(0, len(possible_set)):
                if possible_set[a] == (x, y):
                    return a

    raise NotImplementedError
