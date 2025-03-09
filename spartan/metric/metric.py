import numpy as np

# nodes represent the selected node in each layer.
def evaluator(As:list, Cs:dict, GG:np.ndarray, nodes:list):
    layer = len(As)
    num_of_graphs = len(As) + len(Cs.keys()) // 2
    min_dsty = np.Inf
    geo_avg_dsty = 1
    ari_avg_dsty = 0
    within_dsty = 0
    cross_dsty = 0

    print(f"There are total {num_of_graphs} graphs in this multilayered network.")
    print("Size of each layer: ", [len(node) for node in nodes])
    print("\n")
    print("Edge density of each network: ")
    
    for i in range(layer):
        edge_dsty = 0
        if len(nodes[i]) >= 2: # at least 2 nodes
            edge_dsty = As[i][nodes[i],:][:,nodes[i]].sum() / (len(nodes[i])*(len(nodes[i])-1))
        min_dsty = min(min_dsty, edge_dsty)
        geo_avg_dsty *= edge_dsty
        ari_avg_dsty += edge_dsty
        within_dsty += edge_dsty

        print("Within-layer {}, the edge density is {:.4g}".format(i+1, edge_dsty))

        for j in range(i+1, layer):
            if GG[i,j] != 0:
                edge_dsty = 0
                # if len(nodes[i]) >= 1 and len(nodes[j]) >= 1:
                if len(nodes[i]) >= 2 and len(nodes[j]) >= 2:
                    edge_dsty = Cs[(i,j)][nodes[i],:][:,nodes[j]].sum() / (len(nodes[i])*len(nodes[j]))
                min_dsty = min(min_dsty, edge_dsty)
                geo_avg_dsty *= edge_dsty
                ari_avg_dsty += edge_dsty
                cross_dsty += edge_dsty
                print("Cross-layer {}, the edge density is {:.4g}".format((i+1,j+1),edge_dsty))

    geo_avg_dsty = np.power(geo_avg_dsty, 1/num_of_graphs) # geometric mean
    ari_avg_dsty /= num_of_graphs                         # arithmetic mean
    mixed_dsty = within_dsty*cross_dsty 

    print("\n")
    print("We define some metrics to measure the quality of the dense multilayer subgraph :")
    print("Minimum edge density is {:.4g}".format(min_dsty))
    print("Geometric average edge density is {:.4g}".format(geo_avg_dsty))
    print("Arithmetical average edge density is {:.4g}".format(ari_avg_dsty))
    print("Mixed edge density is {:.4g}".format(mixed_dsty))


    return min_dsty, geo_avg_dsty, ari_avg_dsty, mixed_dsty

def EvaluateF1(mined_res:set, ground_truth:set):
    # Input are two sets of nodes.
    tmp = ground_truth & mined_res
    precision = len(tmp) / len(mined_res) if len(mined_res) else 0
    recall = len(tmp) / len(ground_truth)

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2*precision*recall/(precision+recall)
    return precision, recall, f1


def evaluate_degree_density(As: list, Cs: dict, GG: np.ndarray, nodes: list):
    layer = len(As)
    count = 0
    min_density = np.Inf
    geo_density = 1
    avg_density = 0
    within = 0
    cross = 0

    print("Size of each layer: ", [len(node) for node in nodes])
    print("Average degree density of each network: ")

    for i in range(layer):
        degree_density = 0
        if len(nodes[i]) >= 2:
            degree_density = As[i][nodes[i], :][:, nodes[i]].nnz / (len(nodes[i])*2)
        min_density = min(min_density, degree_density)
        geo_density *= degree_density
        avg_density += degree_density
        within += degree_density
        count += 1

        print("Within-layer {}, the average degree density is {:.4g}".format(i+1,degree_density))
        for j in range(layer):
            if j > i and GG[i, j] != 0:
                degree_density = 0
                if len(nodes[i]) >= 2 and len(nodes[j]) >= 2:
                    degree_density = Cs[(i, j)][nodes[i], :][:, nodes[j]
                                                      ].nnz / np.sqrt((len(nodes[i])*len(nodes[j])))
                min_density = min(min_density, degree_density)
                geo_density *= degree_density
                avg_density += degree_density
                cross += degree_density
                count += 1
                print("Cross-layer {}, the average degree density is {:.4g}".format((i+1,j+1),degree_density))
                
    geo_density = np.power(geo_density,1/count)
    avg_density /= count
    cross_density = within*cross

    print("\n")
    print("Our defined metric to measure the quality of the mined result:")
    print("Minimum edge density is {:.4g}".format(min_density))
    print("Geometric mean edge density is {:.4g}".format(geo_density))
    print("Average mean edge density is {:.4g}".format(avg_density))
    print("Cross edge density is {:.4g}".format(cross_density))


    return min_density, geo_density,avg_density,cross_density
