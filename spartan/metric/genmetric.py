import numpy as np


# compute score as sum of 1- and 2- term interactions (currently just sum of matrix entries)
def c2Score(M, rowSet, colSet):
    return M[list(rowSet),:][:,list(colSet)].sum(axis=None)


def jaccard(pred, actual):
    intersectSize = len(set.intersection(pred[0], actual[0])) + len(set.intersection(pred[1], actual[1]))
    unionSize = len(set.union(pred[0], actual[0])) + len(set.union(pred[1], actual[1]))
    return intersectSize / unionSize

def getPrecision(pred, actual):
    intersectSize = len(set.intersection(pred[0], actual[0])) + len(set.intersection(pred[1], actual[1]))
    return intersectSize / (len(pred[0]) + len(pred[1]))

def getRecall(pred, actual):
    intersectSize = len(set.intersection(pred[0], actual[0])) + len(set.intersection(pred[1], actual[1]))
    return intersectSize / (len(actual[0]) + len(actual[1]))

def getFMeasure(pred, actual):
    prec = getPrecision(pred, actual)
    rec = getRecall(pred, actual)
    return 0 if (prec + rec == 0) else (2 * prec * rec / (prec + rec))

def getRowPrecision(pred, actual, idx):
    intersectSize = len(set.intersection(pred[idx], actual[idx]))
    return intersectSize / len(pred[idx])

def getRowRecall(pred, actual, idx):
    intersectSize = len(set.intersection(pred[idx], actual[idx]))
    return intersectSize / len(actual[idx])

def getRowFMeasure(pred, actual, idx):
    prec = getRowPrecision(pred, actual, idx)
    rec = getRowRecall(pred, actual, idx)
    return 0 if (prec + rec == 0) else (2 * prec * rec / (prec + rec))