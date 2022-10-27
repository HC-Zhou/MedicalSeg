# -*- coding:utf-8  -*-
"""
Time: 2022/8/4 14:42
Author: Yimohanser
Software: PyCharm
"""
from lib.utils.metrics import dc, jc, precision, recall


def get_score(pred, gt):
    p = precision(pred, gt)
    r = recall(pred, gt)
    DICE = dc(pred, gt)
    IoU = jc(pred, gt)
    return {'precision': p, 'recall': r,
            'Dice': DICE,
            'IoU': IoU}


if __name__ == '__main__':
    import numpy as np
    pred = np.ones((1, 3, 3))
    gt = np.array([[[1, 1, 0],
                    [1, 0, 1],
                    [0, 1, 1]]])
    metric = get_score(pred=pred, gt=gt)
    print(metric)
    for key, value in metric.items():
        print(key, value)