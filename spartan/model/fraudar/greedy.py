# contains functions that run the greedy detector for dense regions in a sparse matrix.
# use aveDegree or sqrtWeightedAveDegree or logWeightedAveDegree on a sparse matrix,
# which returns ((rowSet, colSet), score) for the most suspicious block.

from __future__ import division

import numpy as np
# from sklearn.decomposition import TruncatedSVD
from scipy import sparse
from ...util.MinTree import MinTree

from ...metric.genmetric import c2Score

# np.set_printoptions(threshold=numpy.nan)
np.set_printoptions(linewidth=160)

# given a list of lists where each row is an edge, this returns the sparse matrix representation of the data.
# @profile
def listToSparseMatrix(edgesSource, edgesDest):
    m = max(edgesSource) + 1
    n = max(edgesDest) + 1
    M = sparse.coo_matrix(([1]*len(edgesSource), (edgesSource, edgesDest)), shape=(m, n))
    M1 = M > 0
    return M1.astype('int')

# reads matrix from file and returns sparse matrix. first 2 columns should be row and column indices
# of ones.
# @profile
def readData(filename):
    # dat = np.genfromtxt(filename, delimiter='\t', dtype=int)
    edgesSource = []
    edgesDest = []
    with open(filename) as f:
        for line in f:
            toks = line.split()
            edgesSource.append(int(toks[0]))
            edgesDest.append(int(toks[1]))
    return listToSparseMatrix(edgesSource, edgesDest)


def detectMultiple(M, detectFunc, numToDetect):
    Mcur = M.copy().tolil()
    res = []
    for i in range(numToDetect):
        ((rowSet, colSet), score) = detectFunc(Mcur)
        res.append(((rowSet, colSet), score))
        (rs, cs) = Mcur.nonzero()
        for i in range(len(rs)):
            if rs[i] in rowSet and cs[i] in colSet:
                Mcur[rs[i], cs[i]] = 0
    return res


def sqrtWeightedAveDegree(M, maxsize=-1):
    m, n = M.shape
    colSums = M.sum(axis=0)
    colWeights = 1.0 / np.sqrt(np.squeeze(colSums.A) + 5)
    colDiag = sparse.lil_matrix((n, n))
    colDiag.setdiag(colWeights)
    W = M * colDiag
    return fastGreedyDecreasing(W, colWeights, maxsize)

def logWeightedAveDegree(M, maxsize=-1):
    (m, n) = M.shape
    colSums = M.sum(axis=0)
    colWeights = 1.0 / np.log(np.squeeze(colSums.A) + 5)
    colDiag = sparse.lil_matrix((n, n))
    colDiag.setdiag(colWeights)
    W = M * colDiag
    print("finished computing weight matrix")
    return fastGreedyDecreasing(W, colWeights, maxsize)


def aveDegree(M, maxsize=-1):
    m, n = M.shape
    return fastGreedyDecreasing(M, [1] * n, maxsize)


# @profile
def fastGreedyDecreasing(M, colWeights, maxsize=-1):
    (m, n) = M.shape
    Md = M.todok()
    Ml = M.tolil()
    Mlt = M.transpose().tolil()
    rowSet = set(range(0, m))
    colSet = set(range(0, n))
    curScore = c2Score(M, rowSet, colSet)
    bestAveScore = curScore / (len(rowSet) + len(colSet))
    bestSets = (rowSet, colSet)
    print("finished setting up greedy")
    updated = False
    print("InitAveScore: %s with shape (%d, %d)" % (bestAveScore, len(rowSet), len(colSet)))
    rowDeltas = np.squeeze(M.sum(axis=1).A) # *decrease* in total weight when *removing* this row
    colDeltas = np.squeeze(M.sum(axis=0).A)
    print("finished setting deltas")
    rowTree = MinTree(rowDeltas)
    colTree = MinTree(colDeltas)
    print("finished building min trees")

    numDeleted = 0
    deleted = []
    bestNumDeleted = 0

    while rowSet and colSet:
        if (len(colSet) + len(rowSet)) % 100000 == 0:
            print("current set size = ", len(colSet) + len(rowSet))
        (nextRow, rowDelt) = rowTree.getMin()
        (nextCol, colDelt) = colTree.getMin()
        if rowDelt <= colDelt:
            curScore -= rowDelt
            for j in Ml.rows[nextRow]:
                delt = colWeights[j]
                colTree.changeVal(j, -colWeights[j])
            rowSet -= {nextRow}
            rowTree.changeVal(nextRow, float('inf'))
            deleted.append((0, nextRow))
        else:
            curScore -= colDelt
            for i in Mlt.rows[nextCol]:
                delt = colWeights[nextCol]
                rowTree.changeVal(i, -colWeights[nextCol])
            colSet -= {nextCol}
            colTree.changeVal(nextCol, float('inf'))
            deleted.append((1, nextCol))

        numDeleted += 1
        curAveScore = curScore / (len(colSet) + len(rowSet))

        if curAveScore > bestAveScore or (not updated):
            is_update = False
            if isinstance(maxsize, (int, float)):
                if (maxsize==-1 or (maxsize >= len(rowSet) + len(colSet))):
                    is_update = True
            elif maxsize[0]>=len(rowSet) and maxsize[1]>=len(colSet):
                is_update = True

            if is_update:
                updated = True
                bestAveScore = curAveScore
                bestNumDeleted = numDeleted
            

    # reconstruct the best row and column sets
    finalRowSet = set(range(m))
    finalColSet = set(range(n))
    for i in range(bestNumDeleted):
        if deleted[i][0] == 0:
            finalRowSet.remove(deleted[i][1])
        else:
            finalColSet.remove(deleted[i][1])
    return (finalRowSet, finalColSet, bestAveScore)



