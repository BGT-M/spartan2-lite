#!/usr/bin/env python3
# -*- coding:utf-8 -*-

class TopDownEdge:
    graph = None
    graph_size = 0
    v_est = None
    core_num = None
    index = None
    list_ = None
    
    def __init__(self, graph, graph_size):
        self.graph = graph
        self.graph_size = graph_size
        
        self.v_est = [0] * graph_size
        self.core_num = [-1] * graph_size
        self.index = [0] * graph_size
    
    def estimate(self):
        self.v_est = [len(self.graph[i]) for i in range(self.graph_size)]
        # max_val = max(self.v_est)

        for i in range(self.graph_size):
            arr = list()
            for n_i in self.graph[i]:
                if self.v_est[n_i] < self.v_est[i]:
                    arr.append(self.v_est[n_i])

            arr = sorted(arr)
            max_min = self.v_est[i]
            for j in range(len(arr)):
                temp = max(arr[j], len(self.graph[i]) - j - 1)
                if max_min > temp:
                    max_min = temp
            
            if max_min > 0:
                self.v_est[i] = max_min
    
    def compute_core(self, k_l, k_u):
        k = k_l
        over = False
        while k <= k_u:
            while True:
                condition = True
                i = 0
                while i < len(self.list_):
                    t_i = self.list_[i]
                    if self.index[t_i] < k:
                        condition = False
                        # print(f" *** remove {i}")
                        del self.list_[i]
                        i -= 1
                        self.v_est[t_i] = k - 1
                        
                        # self.core_num[t_i] = -1
                        # self.index[t_i] = 0
                        
                        for v in self.graph[t_i]:
                            if self.core_num[v] >= 0:
                                self.index[v] -= 1
                    i += 1
                if condition:
                    break
            
            for u in self.list_:
                self.core_num[u] = k
                over = True
            k += 1
        
        return over

    def EMCore(self, k_l, k_u):
        self.list_ = list()
    
        for i in range(self.graph_size):
            if self.v_est[i] >= k_l:
                self.core_num[i] = 0
                self.list_.append(i)
        
        for u in self.list_:
            count = 0
            for v in self.graph[u]:
                if self.core_num[v] >= 0:
                    count += 1
            
            self.index[u] = count
        
        print("k_u ", k_u)
        over = self.compute_core(k_l, k_u)
        core = k_l if over else -1

        return core

    def TDAlg(self):
        self.estimate()
        k_u = max(self.v_est)
        
        while True:
            count = 0
            for i in range(self.graph_size):
                if k_u <= self.v_est[i]:
                    self.v_est[i] = k_u
                    count += 1
            
            if count * (count - 1) / 2 > k_u:
                break
            k_u -= 1
        
        core = -1
        k_l = k_u
        while True:
            core = self.EMCore(k_l, k_u)
            k_u = k_l - 1
            k_l = k_u
            
            if core != -1:
                break
