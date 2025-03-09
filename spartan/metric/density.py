from scipy.sparse import csr_matrix

def avg_degree_density(adj:csr_matrix, nids:list):
    num_nodes = len(nids)
    num_edges = adj[nids,:][:,nids].nnz // 2
    return num_edges / num_nodes

    
def clique_density(adj:csr_matrix, nids:list):
    num_nodes = len(nids)
    num_edges = adj[nids,:][:,nids].nnz // 2
    return num_edges / (num_nodes * (num_nodes-1) / 2)


def edge_surplus_density(adj:csr_matrix, nids:list, alpha:float):
    num_nodes = len(nids)
    num_edges = adj[nids,:][:,nids].nnz // 2
    return num_edges - alpha * (num_nodes * (num_nodes-1) / 2)