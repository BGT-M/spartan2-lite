#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np

from .kcore import KCore

class KList:
    graph = None
    graph_size = 0
    gen_graph = None
    
    deg = None
    label = None
    order = None
    
    k = 0
    motif_num = 0
    motif_deg = None
    statistic = None
    adjlist = None

    def __init__(self, graph, k):
        self.gen_graph = graph
        self.k = k
        self.graph_size = len(graph)
        self.graph = [[] * i for i in range(self.graph_size)]
        self.order = [0] * self.graph_size
        self.deg = [0] * self.graph_size
        self.label = [k] * self.graph_size
        self.motif_deg = [0] * self.graph_size
        self.statistic = dict()

    def get_motif_deg(self):
        return self.motif_deg

    def get_motif_num(self):
        return self.motif_num

    def list_fast(self):
        self.get_listing_order()
        self.generate_dag()
        c_list = list()
        arr = list(np.arange(self.graph_size))
        self.__listing__(self.k, c_list, arr)

    def list_record(self):
        self.get_listing_order()
        self.generate_dag()
        c_list = list()
        arr = list(np.arange(self.graph_size))
        self.__listing_record__(self.k, c_list, arr)

    def list_adj(self):
        self.get_listing_order()
        self.generate_dag()
        c_list = list()
        arr = list(np.arange(self.graph_size))

        self.adjlist = [[] * i for i in range(self.graph_size)]
        self.__listing_adj__(self.k, c_list, arr)

    def list_one(self, elem):
        self.get_listing_order()
        self.generate_dag()
        c_list = list()
        arr = list(np.arange(self.graph_size))
        self.__listing_elem__(self.k, c_list, arr, elem)

    def list_batch(self, elems):
        self.get_listing_order()
        self.generate_dag()
        c_list = list()
        arr = list(np.arange(self.graph_size))
        self.__listing_elements__(self.k, c_list, arr, elems)

    def get_listing_order(self):
        k_cores = KCore(self.gen_graph)
        k_cores.decompose()
        arr = k_cores.get_core_reverse()
        for i in range(self.graph_size):
            self.order[arr[i]] = i + 1

    def generate_dag(self):
        for i in range(self.graph_size):
            cnt = 0
            arr = list()
            for j in self.gen_graph[i]:
                if self.order[i] < self.order[j]:
                    arr.append(j)
                    cnt += 1
            self.deg[i] = cnt
            self.graph[i] = arr

    def __listing_record__(self, k, c_list, arr):
        if k == 2:
            info = " ".join(map(str, c_list))
            multi = 0
            for u in arr:
                for j in range(self.deg[u]):
                    v = self.graph[u][j]
                    info += str(u) + " " + str(v)
                    multi += 1
                    self.motif_num += 1
                    self.motif_deg[v] += 1
                    self.motif_deg[u] += 1

                    temp_arr = [0] * (self.k + 1)
                    for m in range(len(c_list)):
                        temp_arr[m] = c_list[m]
                    temp_arr[self.k - 2] = u
                    temp_arr[self.k - 1] = v
                    temp_arr[self.k] = 1
                    self.statistic[info] = temp_arr

            for m in c_list:
                self.motif_deg[m] += multi
        else:
            for u in arr:
                arr_n = list()
                for v in self.graph[u]:
                    if self.label[v] == k:
                        self.label[v] = k - 1
                        arr_n.append(v)

                for v_t in arr_n:
                    index = 0
                    for m in range(len(self.graph[v_t]) - 1, index, -1):
                        if self.label[self.graph[v_t][m]] == k - 1:
                            while index < m and self.label[self.graph[v_t][index]] == k - 1:
                                index += 1
                            if self.label[self.graph[v_t][index]] != k - 1:
                                w = self.graph[v_t][m]
                                self.graph[v_t][m] = self.graph[v_t][index]
                                self.graph[v_t][index] = w

                    if len(self.graph[v_t]) != 0 and self.label[self.graph[v_t][index]] == k - 1:
                        index += 1
                    self.deg[v_t] = index

                c_list.append(u)
                self.__listing_record__(k - 1, c_list, arr_n)
                c_list.pop(-1)

                for v_t in arr_n:
                    self.label[v_t] = k

    def __listing_adj__(self, k, c_list, arr):
        if k == 2:
            info = " ".join(map(str, c_list))
            multi = 0
            for u in arr:
                for j in range(self.deg[u]):
                    v = self.graph[u][j]
                    info_v = info + f" {u} {v}"
                    multi += 1
                    self.motif_num += 1
                    self.motif_deg[v] += 1
                    self.motif_deg[u] += 1

                    temp_arr =[0] * (self.k + 1)
                    for m in range(len(c_list)):
                        temp_arr[m] = c_list[m]

                    temp_arr[self.k - 2] = u
                    temp_arr[self.k - 1] = v
                    temp_arr[self.k] = 1
                    self.statistic[info_v] = temp_arr

                    for m in range(self.k):
                        self.adjlist[temp_arr[m]].append(info_v)

            for m in c_list:
                self.motif_deg[m] += multi
        else:
            for u in arr:
                arr_n = list()
                for v in self.graph[u]:
                    if self.label[v] == k:
                        self.label[v] = k - 1
                        arr_n.append(v)

                for v_t in arr_n:
                    index = 0
                    for m in range(len(self.graph[v_t]) - 1, index, -1):
                        if self.label[self.graph[v_t][m]] == k - 1:
                            while index < m and self.label[self.graph[v_t][index]] == k - 1:
                                index += 1
                            if self.label[self.graph[v_t][index]] != k - 1:
                                w = self.graph[v_t][m]
                                self.graph[v_t][m] = self.graph[v_t][index]
                                self.graph[v_t][index] = w

                    if len(self.graph[v_t]) != 0 and self.label[self.graph[v_t][index]] == k - 1:
                        index += 1
                    self.deg[v_t] = index

                c_list.append(u)
                self.__listing_adj__(k - 1, c_list, arr_n)
                c_list.pop(-1)

                for v_t in arr_n:
                    self.label[v_t] = k

    def __listing__(self, k, c_list, arr):
        if k == 2:
            info = " ".join(map(str, c_list))
            multi = 0
            for u in arr:
                for j in range(self.deg[u]):
                    v = self.graph[u][j]
                    # print(info + f" {u} {v}")
                    multi += 1
                    self.motif_num += 1
                    self.motif_deg[v] += 1
                    self.motif_deg[u] += 1

            for m in c_list:
                self.motif_deg[m] += multi
        else:
            for u in arr:
                arr_n = list()
                for v in self.graph[u]:
                    if self.label[v] == k:
                        self.label[v] = k - 1
                        arr_n.append(v)

                for v_t in arr_n:
                    index = 0
                    for m in range(len(self.graph[v_t]) - 1, index, -1):
                        if self.label[self.graph[v_t][m]] == k - 1:
                            while index < m and self.label[self.graph[v_t][index]] == k - 1:
                                index += 1
                            if self.label[self.graph[v_t][index]] != k - 1:
                                w = self.graph[v_t][m]
                                self.graph[v_t][m] = self.graph[v_t][index]
                                self.graph[v_t][index] = w

                    if len(self.graph[v_t]) != 0 and self.label[self.graph[v_t][index]] == k - 1:
                        index += 1
                    self.deg[v_t] = index

                c_list.append(u)
                self.__listing__(k - 1, c_list, arr_n)
                c_list.pop(-1)

                for v_t in arr_n:
                    self.label[v_t] = k

    def __listing_elem__(self, k, c_list, arr, elem):
        if k == 2:
            one_node = elem in c_list
            info = " ".join(map(str, c_list))
            multi = 0
            for u in arr:
                for j in range(self.deg[u]):
                    v = self.graph[u][j]
                    print(info + f" {u} {v}")
                    if one_node or u == elem or v == elem:
                        multi += 1
                        self.motif_num += 1
                        self.motif_deg[v] += 1
                        self.motif_deg[u] += 1

            for m in c_list:
                self.motif_deg[m] += multi
        else:
            for u in arr:
                arr_n = list()
                for v in self.graph[u]:
                    if self.label[v] == k:
                        self.label[v] = k - 1
                        arr_n.append(v)

                for v_t in arr_n:
                    index = 0
                    for m in range(len(self.graph[v_t]) - 1, index, -1):
                        if self.label[self.graph[v_t][m]] == k - 1:
                            while index < m and self.label[self.graph[v_t][index]] == k - 1:
                                index += 1
                            if self.label[self.graph[v_t][index]] != k - 1:
                                w = self.graph[v_t][m]
                                self.graph[v_t][m] = self.graph[v_t][index]
                                self.graph[v_t][index] = w

                    if len(self.graph[v_t]) != 0 and self.label[self.graph[v_t][index]] == k - 1:
                        index += 1
                    self.deg[v_t] = index

                c_list.append(u)
                self.__listing_elem__(k - 1, c_list, arr_n, elem)
                c_list.pop(-1)

                for v_t in arr_n:
                    self.label[v_t] = k

    def __listing_elements__(self, k, c_list, arr, elems_vect):
        if k == 2:
            one_node = False
            for m in c_list:
                if elems_vect[m] == 1:
                    one_node = True
                    break
            
            info = " ".join(map(str, c_list))
            multi = 0
            for u in arr:
                for j in range(self.deg[u]):
                    v = self.graph[u][j]
                    # print(info + f" {u} {v}")
                    if one_node or elems_vect[u] == 1 or elems_vect[v] == 1:
                        multi += 1
                        self.motif_num += 1
                        self.motif_deg[v] += 1
                        self.motif_deg[u] += 1

            for m in c_list:
                self.motif_deg[m] += multi
        else:
            for u in arr:
                arr_n = list()
                for v in self.graph[u]:
                    # print(f"****{v}  {self.label[v]} {k}" )
                    if self.label[v] == k:
                        self.label[v] = k - 1
                        arr_n.append(v)

                for v_t in arr_n:
                    index = 0
                    for m in range(len(self.graph[v_t]) - 1, index, -1):
                        if self.label[self.graph[v_t][m]] == k - 1:
                            while index < m and self.label[self.graph[v_t][index]] == k - 1:
                                index += 1
                            if self.label[self.graph[v_t][index]] != k - 1:
                                w = self.graph[v_t][m]
                                self.graph[v_t][m] = self.graph[v_t][index]
                                self.graph[v_t][index] = w

                    if len(self.graph[v_t]) != 0 and self.label[self.graph[v_t][index]] == k - 1:
                        index += 1
                    self.deg[v_t] = index

                c_list.append(u)
                self.__listing_elements__(k - 1, c_list, arr_n, elems_vect)
                c_list.pop(-1)

                for v_t in arr_n:
                    self.label[v_t] = k


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

if __name__ == '__main__':
    infn = './baseline/kds/datasets/test.txt'
    
    graph, graph_size = load_adjlist(infn)
    a = [0] * graph_size
    a[33] = 1
    a[855] = 1
    kl = KList(graph, 3)
    kl.list_batch(a)
    motif_deg = kl.get_motif_deg()
    print(motif_deg[33])