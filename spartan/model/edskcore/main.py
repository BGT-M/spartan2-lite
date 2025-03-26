from kcore import KCore
from klist import KList
from scipy.sparse import lil_matrix
from TDCDS import TDCDS

def combination_precom(start_num, max_degree):
    res = [0] * (max_degree + 1)
    res[start_num] = 1
    for i in range(start_num + 1, max_degree+1, 1):
        res[i] = int(res[i - 1] * i * 1.0 / (i - start_num))
    return res

def efficient_core_dsd(graph:lil_matrix):
    graph_size = graph.shape[0]
    graph_adjlist = []
    for i in range(graph_size):
        graph_adjlist.append(graph.rows[i])
    motif_type, motif_count = 2, 2
    motif = [[0, 1], [1, 0]]
    motif_length = len(motif)
    k_core = KCore(graph_adjlist)
    k_core.decompose()
    max_k = k_core.get_maxcore()
    comb_arr = combination_precom(motif_length - 1, max_k)
    td_cds_algo = TDCDS(graph_adjlist, graph_size, motif, motif_count, motif_type)
    td_cds_algo.estimate_by_core(comb_arr)
    td_cds_algo.TDAlg()
    score = td_cds_algo.get_density()
    res = td_cds_algo.get_results()
    return res,score