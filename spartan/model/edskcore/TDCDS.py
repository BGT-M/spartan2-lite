#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np

from .kcore import KCore
from .klist import KList
from .combination import combination_precom

class TDCDS:
    graph = None
    graph_size = 0
    v_est = None
    motif_size = 0
    deposit = 0
    core_num = None
    index = None
    record_index = None
    # motif = None
    motif_type = -1
    motif_d = 0
    motif_graph = None
    
    insert_ = list()
    delete_ = list()
    list_ = list()
    mark = list()
    new_arr = list()
    new_map = list()
    min_core = 0
    combinate = list()
    cur_max = 0
    cur_core = 0

    def __init__(self, graph, graph_size, motif, motif_type, motif_d):
        self.graph = graph
        self.graph_size = graph_size
        self.motif = motif
        self.motif_type = motif_type
        self.motif_d = motif_d
        self.motif_size = len(motif)
        
        self.v_est = [0] * graph_size
        self.index = [0] * graph_size 
        self.core_num = [-1] * graph_size
        self.record_index = [-1] * graph_size 
        
        self.mark = [0] * graph_size 
        self.new_arr = [0] * graph_size 
        self.new_map =[0] * graph_size
        self.delete_ = [0] * graph_size 
        self.insert_ = [0] * graph_size  
    
    def estimate_by_core(self, combination):
        kcore = KCore(self.graph)
        self.combinate = combination
        degs = kcore.decompose()
        
        for i, d_i in enumerate(degs):
            self.v_est[i] = combination[d_i]
    
    def estimate(self, arr):
        self.combinate = arr
        self.motif_graph = dict()
        kk = KList(self.graph, self.motif_size)
        kk.list_fast()
        self.v_est = kk.get_motif_deg()
    
    def get_results(self):
        deposit_nodes = np.where(np.array(self.v_est) == self.deposit)[0]
        # print(deposit_nodes)
        return list(deposit_nodes)

    def get_density(self):
        deposit_nodes = np.where(np.array(self.v_est) == self.deposit)[0]
        n_vert = len(deposit_nodes)
        arr_1 = [0] * self.graph_size
        arr_2 = [-1] * self.graph_size
        arr_3 = [0] * self.graph_size
        arr_4 = [0] * self.graph_size
        
        for i in deposit_nodes:
            arr_2[i] = 0
            arr_3[i] = 1
        nd_map = self.generate(arr_3, deposit_nodes, arr_2, arr_4, arr_1)
        n_clique = 0
        if nd_map is not None:
            n_clique = np.sum(list(nd_map.values()))
        
        n_clique /= self.motif_size
        print(deposit_nodes)
        density = n_clique * 1.0 / n_vert
        print(f"# of clique: {n_clique}: # of vertex: {n_vert}: density: {density}")
        return density
        

    def generate(self, index_arr, c_list, mark_arr, arr, map_s):
        temp_list = list()
        queue = list()
        for e in c_list:
            temp_list.append(e)
            arr[e] = 1
            queue.append(e)
        
        d = 1
        while len(queue) > 0 and d <= 1:
            u = queue.pop(0)
            d = arr[u]
            for v in self.graph[u]:
                if arr[v] == 0 and d <= 1 and mark_arr[v] == 0:
                    queue.append(v)
                    arr[v] = d + 1
                    temp_list.append(v)
        
        num = 0
        count = len(temp_list)
        map_arr = [0] * count
        new_graph = [None] * count
        
        for i in range(count):
            node = temp_list[i]
            map_arr[i] = node
            map_s[node] = num
            
            elem_list = list()
            for u in self.graph[node]:
                if arr[u] != 0 and u != node:
                    elem_list.append(u)
                    
            new_graph[num] = elem_list
            num += 1
        
        for i in range(count):
            update = temp_list[i]
            arr[update] = 0
            
            for j in range(len(new_graph[i])):
                new_graph[i][j] = map_s[new_graph[i][j]]
        
        map_d = [0] * len(new_graph)
        for i in range(len(new_graph)):
            if index_arr[map_arr[i]] == 1:
                map_d[i] = 1
        

        f = KList(new_graph, self.motif_size)
        f.list_batch(map_d)
        t_a = f.get_motif_deg()
        
        res = dict()
        for i in range(len(t_a)):
            # print(t_a[i])
            if t_a[i] > 0:
                res[map_arr[i]] = t_a[i]
            # if map_d[i] == 1:
            #     print(f"{t_a[i]}  {len(new_graph[i])}  {len(new_graph)}  {len(t_a)}  {map_arr[i]}  {i}" )
            
        return res
    
    def generate_one(self, index, mark_arr, arr, map_s):
        temp_list = list([index])
        queue = list()
        queue.append(index)
        arr[index] = 1
        
        d = 1
        while len(queue) > 0 and d <= 1:
            u = queue.pop(0)
            d = arr[u]
            for v in self.graph[u]:
                if arr[v] == 0 and d <= 1 and mark_arr[v] == 0:
                    queue.append(v)
                    arr[v] = d + 1
                    temp_list.append(v)
        
        num = 0
        count = len(temp_list)
        map_arr = [0] * count
        new_graph = [None] * count
        
        for i in range(count):
            node = temp_list[i]
            map_arr[i] = node
            map_s[node] = num
            
            elem_list = list()
            for u in self.graph[node]:
                if arr[u] != 0 and u != node:
                    elem_list.append(u)
                    
            new_graph[num] = elem_list
            num += 1
        
        for i in range(count):
            update = temp_list[i]
            arr[update] = 0
            
            for j in range(len(new_graph[i])):
                new_graph[i][j] = map_s[new_graph[i][j]]

        f = KList(new_graph, self.motif_size)
        f.list_one(0)
        t_a = f.get_motif_deg()
        # print(t_a[0])
        
        res = dict()
        for i in range(1, len(t_a)):
            if t_a[i] > 0:
                res[map_arr[i]] = t_a[i]
            
        return res
    
    
    def compute_core(self, k_l, k_u):
        num = np.sum(np.array(self.list_) >= k_l)
        if num < len(self.combinate) and self.combinate[num] < k_l:
            return False
        
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
                        del self.list_[i]
                        i -= 1
                        # self.list_.remove(i)
                        # i -= 1
                        self.v_est[t_i] = k - 1
                        if self.index[t_i] > 0:
                            map_ti = self.generate_one(t_i, self.mark, self.new_arr, self.new_map)
                            if len(map_ti) > 0:
                                for s, v in map_ti.items():
                                    self.index[s] -= v
                        
                        self.mark[t_i] = -1
                    # else:
                    #     i += 1
                    
                if condition:
                    break
        
            for u in self.list_:
                self.core_num[u] = k
                over = True

            cnt = len(np.where(np.array(self.list_, int) == -1)[0])
            if cnt == len(self.list_): # len(self.list_) <= 0:
                break
            k += 1
        
        self.deposit = k - 1
        return over
    
    def compute_core_batch(self, k_l, k_u):
        k = k_l
        over = False
        while k <= k_u:
            while True:
                condition = True
                list_delete = list()
                i = 0
                while i < len(self.list_):
                    # print(f"## {i}: {len(self.list_)} {self.list_[i]}")
                    t_i = self.list_[i]
                    if self.index[t_i] < k:
                        # if i in self.list_:
                        # print(" ---- remove:", i)
                        del self.list_[i]
                        i -= 1
                        self.delete_[t_i] = 1
                        self.v_est[t_i] = k - 1
                        if self.index[t_i] != 0:
                            list_delete.append(t_i)
                        self.mark[t_i] = -1
                    i += 1

                map_ti = self.generate(self.delete_, list_delete, self.mark, self.new_arr, self.new_map)
                if len(map_ti) > 0:
                    for s, v in map_ti.items():
                        self.index[s] -= v
                
                for u in self.list_:
                    if self.index[u] < k:
                        condition = False
                
                if condition:
                    break
            
            if len(self.list_) > 0:
                k_max = 0xFFFFFF
                for t_i in self.list_:
                    if self.index[t_i] < k_max:
                        k_max = self.index[t_i]
                if k < k_max:
                    k = k_max
            
            if k >= self.cur_max or len(self.list_) > 0:
                over = True
            
            if len(self.list_) == 0:
                break
            
            self.cur_core = k
            k = k + 1

        self.deposit = k - 1
        return over
    
    
    ## ICDE 2011:  Efficient core decomposition in massive networks. 
    def EMCore(self, k_l, k_u):
        min_val = 0xFFFFFF
        
        temp_list = list()
        for u in self.list_:
            self.insert_[u] = 0
            if self.record_index[u] == -1:
                self.mark[u] = 0
                temp_list.append(u)
                self.insert_[u] = 1
        
        map_t = self.generate(self.insert_, temp_list, self.mark, self.new_arr, self.new_map)
        for node, deg in map_t.items():
            if self.record_index[node] == -1:
                self.record_index[node] = deg
            else:
                self.record_index[node] += deg
        
        record_index_ = [self.record_index[u] for u in self.list_]
        k_l = np.min(record_index_)
        k_u = np.max(record_index_)
        k_l = max(self.cur_core, k_l)
        
        is_del = False
        for u in self.list_:
            self.index[u] = self.record_index[u]
            if min_val > self.index[u] and self.index[u] != 0:
                min_val = self.index[u]
                is_del = True
        
        # if not is_del:
        #     min_val = 0
        # if again == 0 and k_l > self.min_core:
        #     return -1

        if self.min_core < min_val:
            self.min_core = min_val
        # print(f"k_u: {k_u}, current_max: {self.cur_max},  current_core: {self.cur_core}")
        
        print(f"****  compute core batch: {k_l - 1} {k_u}")
        over = self.compute_core_batch(k_l - 1, k_u)
        core = k_l if over else -1
        
        return core


    def TDAlg(self):
        self.index = [0] * self.graph_size
        self.mark = [-1] * self.graph_size
        
        core = -1
        iters = 100
        k_l = np.max(self.v_est)
        # k_u = np.max(self.v_est)
        arr = [-1] * self.graph_size
        
        while True:
            k_u = k_l
            max_val = 0
            self.list_ = list()
            while True:
                max_val = 0
                for i in range(self.graph_size):
                    if max_val < self.v_est[i] and arr[i] == -1:
                        max_val = self.v_est[i]
                k_l = max_val
                self.list_ = list()
                for i in range(self.graph_size):
                    if self.v_est[i] >= k_l:
                        arr[i] = 0
                        self.core_num[i] = 0
                        self.list_.append(i)
                        if self.record_index[i] != -1:
                            self.mark[i] = 0
                if len(self.list_) >= min(iters, self.graph_size):
                    break
            
            iters = len(self.list_) * 2
            self.cur_max = 0
            for i in range(self.graph_size):
                if self.v_est[i] > self.cur_max and self.v_est[i] < max_val:
                    self.cur_max = self.v_est[i]
            
            print(f"****  EMCore: {k_l} {k_u}")
            core = self.EMCore(k_l, k_u)
            
            # k_u = k_l - 1
            # k_l = k_u
            # for i in range(self.graph_size):
            #     if max_val < self.V_Est[i] and self.record_index[i] == -1:
            #         max = self.V_Est[i]
            
            if core != -1:
                break
        
        print("max-core: ", self.deposit)
        
def load_adjlist(info, segment=' '):
    graph_size = 0
    graph = None
    n_es = 0
    with open(info, 'r') as fp:
        line = fp.readline().strip()
        graph_size = int(line)
        graph = [[] * i for i in range(graph_size)]
        for line in fp.readlines():
            toks = line.strip().split(segment)
            toks = list(map(int, toks))
            graph[toks[0]] = toks[1:]
            n_es += len(toks[1:])
        fp.close()
    
    print(f"###{n_es // 2}")
    return graph, graph_size

if __name__ == "__main__":
    infn = 'test.txt'
    graph, graph_size = load_adjlist(infn)
    
    motif_type, motif_count = 2, 2
    motif = [[0, 1], [1, 0]]
    motif_length = len(motif)
    
    index = 2
    max_deg = np.max([len(graph[i]) for i in range(graph_size)])
    combinations = list()
    combinations.append(combination_precom(index, max_deg))
    combinations.append(combination_precom(index - 1, max_deg))
    
    if index - 2 == 0:
        combinations.append([1] * max_deg)
    else:
        combinations.append(combination_precom(index - 2, max_deg))
    
    td_cds_algo = TDCDS(graph, graph_size, motif, motif_count, motif_type)
    td_cds_algo.TDAlg()
    td_cds_algo.get_density()