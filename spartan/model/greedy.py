import numpy as np
from ..util.MinTree import MinTree
from scipy.sparse import lil_matrix

""" The input graph should be a unipartite and undirected graph.
"""

def greedy_monopartite(graph:lil_matrix):
    assert graph.shape[0] == graph.shape[1], "Not unipartite graph!"
    edges = graph.sum() / 2
    nodeset = set(range(0, graph.shape[1]))
    cur_score = edges / len(nodeset)
    best_score = 0
    deltas = np.squeeze(graph.sum(axis=1).A)
    tree = MinTree(deltas)
    num_deleted = 0
    deleted = []
    best_num_deleted = 0
    while len(nodeset) > 1:
        node, val = tree.getMin()
        edges -= val
        for j in graph.rows[node]:
            delt = graph[node, j]
            tree.changeVal(j, -delt)
        nodeset -= {node}
        tree.changeVal(node, float('inf'))
        deleted.append(node)
        num_deleted += 1
        cur_score = edges / len(nodeset)
        if cur_score > best_score:
            best_score = cur_score
            best_num_deleted = num_deleted
    res = set(range(0, graph.shape[1]))
    res -= set(deleted[0:best_num_deleted])
    return sorted(res), best_score
            

    

def greeedy_bipartite(graph:lil_matrix):
    (m, n) = graph.shape
    Ml = graph.tolil()
    Mlt = graph.transpose().tolil()
    rowSet = set(range(0, m))
    colSet = set(range(0, n))
    curScore = graph[list(rowSet), :][:, list(colSet)].sum(axis=None)
    bestAveScore = curScore / (len(rowSet) + len(colSet))
    bestSets = (rowSet, colSet)

    # *decrease* in total weight when *removing* this row
    # Prepare the min priority tree to begin greedy algorithm.
    rowDeltas = np.squeeze(graph.sum(axis=1).A)
    colDeltas = np.squeeze(graph.sum(axis=0).A)

    rowTree = MinTree(rowDeltas)
    colTree = MinTree(colDeltas)

    numDeleted = 0
    deleted = []
    bestNumDeleted = 0

    while rowSet and colSet:

        nextRow, rowDelt = rowTree.getMin()
        nextCol, colDelt = colTree.getMin()

        if rowDelt <= colDelt:
            curScore -= rowDelt
            for j in Ml.rows[nextRow]:
                delt = Ml[nextRow, j]
                colTree.changeVal(j, -delt)
            rowSet -= {nextRow}
            rowTree.changeVal(nextRow, float('inf'))
            deleted.append((0, nextRow))
        else:
            curScore -= colDelt
            for i in Mlt.rows[nextCol]:
                delt = Ml[i, nextCol]
                rowTree.changeVal(i, -delt)
            colSet -= {nextCol}
            colTree.changeVal(nextCol, float('inf'))
            deleted.append((1, nextCol))

        numDeleted += 1
        curAveScore = curScore / (len(colSet) + len(rowSet))
        if curAveScore > bestAveScore:
            bestAveScore = curAveScore
            bestNumDeleted = numDeleted

    # Reconstruct the best row and column sets
    finalRowSet = set(range(m))
    finalColSet = set(range(n))
    for i in range(bestNumDeleted):
        if deleted[i][0] == 0:
            finalRowSet.remove(deleted[i][1])
        else:
            finalColSet.remove(deleted[i][1])
    return (finalRowSet, finalColSet, bestAveScore)