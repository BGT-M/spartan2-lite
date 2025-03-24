import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from typing import Tuple, List, Set
from abc import abstractmethod
from .._model import DMmodel

class ExactDSD(DMmodel):
    """Exact densest subgraph detection using Goldberg's max-flow algorithm."""
    
    def __init__(self, data: csr_matrix, **kwargs):
        """
        Parameters
        ----------
        data : csr_matrix
            Sparse adjacency matrix of the graph
        kwargs : dict
            Additional parameters for DMmodel
        """
        super().__init__(data, **kwargs)
        self.adj = data
        self.graph = nx.from_scipy_sparse_matrix(data)
        self._validate()
        
    @classmethod
    def __create__(cls, data: csr_matrix, **kwargs) -> 'ExactDSD':
        """Factory method implementation"""
        return cls(data, **kwargs)
    
    def run(self, **kwargs) -> Tuple[List[int], float]:
        """Execute the densest subgraph detection.
        
        Returns
        -------
        Tuple[List[int], float]
            (nodes of densest subgraph, achieved density)
        """
        return self._find_densest_subgraph()
    
    def _validate(self) -> None:
        """Ensure the graph meets algorithm requirements."""
        if nx.is_directed(self.graph):
            raise ValueError("Graph must be undirected")
    
    def _find_densest_subgraph(self) -> Tuple[List[int], float]:
        """Core implementation of Goldberg's algorithm."""
        m = self.graph.number_of_edges()
        n = self.graph.number_of_nodes()
        
        # Initialize binary search bounds
        min_dst = m / n  # Lower bound (whole graph density)
        max_dst = (n - 1) / 2  # Upper bound (clique density)
        
        opt_dst = 0
        best_res = list(self.graph.nodes())
        
        if min_dst == max_dst:
            return best_res, max_dst
            
        # Binary search loop
        while max_dst - min_dst > 1 / n**2:
            query_dst = (max_dst + min_dst) / 2
            flow_net = self._create_flow_net(query_dst)
            
            # Get minimum cut
            _, (source_partition, _) = nx.minimum_cut(
                flow_net, 's', 't', capacity='capacity'
            )
            
            if source_partition == {'s'}:
                max_dst = query_dst
            else:
                min_dst = query_dst
                best_res = list(source_partition - {'s'})
                opt_dst = query_dst
        
        return best_res, opt_dst
    
    def _create_flow_net(self, query_dst: float) -> nx.DiGraph:
        """Construct the flow network for a given density query."""
        m = self.graph.number_of_edges()
        flow_net = nx.DiGraph()
        
        # Add original edges with capacity 1
        for u, v in self.graph.edges():
            flow_net.add_edge(u, v, capacity=1)
            flow_net.add_edge(v, u, capacity=1)  # Undirected to directed
        
        # Add source and sink connections
        for v in self.graph.nodes():
            flow_net.add_edge('s', v, capacity=m)
            flow_net.add_edge(v, 't', capacity=m + 2*query_dst - self.graph.degree(v))
            
        return flow_net
    
    def anomaly_detection(self) -> List[int]:
        """Interface for anomaly detection (returns nodes of densest subgraph)."""
        nodes, _ = self.run()
        return nodes