import numpy as np
from scipy.sparse import lil_matrix,csc_matrix
from ...util import MinTree

""" The input graph should be symmetric.
    The adj[i,j] should be 0 or 1.
"""

def greedy_oqc(graph:lil_matrix, alpha:float=1/3):
    # Initialize.
    num_nodes = graph.shape[0]
    num_edges = graph.nnz // 2
    cur_set = set(range(num_nodes))
    cur_size = len(cur_set)
    cur_score = num_edges - alpha * cur_size * (cur_size - 1) / 2 
    best_score = cur_score
    
    # Construct min-tree.
    priority = np.squeeze(graph.sum(axis=1).A) # Node's degree as priority
    tree = MinTree(priority)
    
    # Greedy peeling process.
    num_deleted, best_num_deleted = 0, 0
    deleted_nodes = []
    while len(cur_set) > 2:
        min_nid, min_pri = tree.get_min()
        num_edges -= min_pri
        
        # Update its neighbors' priority.
        for neigh in graph.rows[min_nid]:
            pri = graph[min_nid, neigh]
            tree.change_val(neigh, -pri)
        
        # Remove the node.
        cur_set.remove(min_nid)
        tree.change_val(min_nid, float('inf'))
        deleted_nodes.append(min_nid)
        num_deleted += 1
        cur_size -= 1
        cur_score = num_edges - alpha * cur_size * (cur_size - 1) / 2 
        
        if cur_score > best_score:
            best_score = cur_score
            best_num_deleted = num_deleted

    # Reconstruct the best result.
    final_set = set(range(num_nodes)) - set(deleted_nodes[:best_num_deleted])
    res = sorted(final_set)
    return res, best_score

### Todo: (1) localsearch_oqc (2) constrained_oqc




