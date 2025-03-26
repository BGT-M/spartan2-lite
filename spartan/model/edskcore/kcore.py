#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np


class KCore:
    graph = None
    num_nodes = 0
    deg = None
    core_reverse = None

    def __init__(self, graph):
        self.graph = graph
        self.num_nodes = len(graph)
        self.core_reverse = [0] * self.num_nodes

    def decompose(self):
        self.deg = [len(self.graph[i]) for i in range(self.num_nodes)]
        max_deg = -1
        if len(self.deg) > 0:
            max_deg = np.max(self.deg)

        ## fill the bin
        bin = [0] * (max_deg + 1)
        for i in range(self.num_nodes):
            bin[self.deg[i]] += 1
        ## accumulate counts in the bin
        start = 0
        for i in range(max_deg + 1):
            b_i = bin[i]
            bin[i] = start
            start += b_i

        ## find the position
        position = [0] * (self.num_nodes + 1)
        vertex = [0] * (self.num_nodes + 1)
        for v in range(self.num_nodes):
            position[v] = bin[self.deg[v]]
            vertex[position[v]] = v
            bin[self.deg[v]] += 1

        for d in np.arange(max_deg, 0, -1):
            bin[d] = bin[d - 1]

        if len(bin) > 0:
            bin[0] = 1

        ## decompose
        for i in range(self.num_nodes):
            v = vertex[i]
            for u in self.graph[v]:
                if self.deg[u] > self.deg[v]:
                    du, pu = self.deg[u], position[u]
                    pw = bin[du]
                    w = vertex[pw]
                    if u != w:
                        position[u] = pw
                        vertex[pu] = w
                        position[w] = pu
                        vertex[pw] = u
                    bin[du] += 1
                    self.deg[u] -= 1

            self.core_reverse[self.num_nodes - i - 1] = v

        return self.deg

    def get_maxcore(self):
        return np.max(self.deg)

    def get_core_reverse(self):
        return self.core_reverse

    def info(self, verbose=1):
        iters = 1
        size = 5
        cnt = 0

        info_str = ""
        while True:
            cnt = 0
            for i in range(len(self.deg)):
                if self.deg[i] < size * iters and self.deg[i] >= size * (iters - 1):
                    cnt += 1

            if cnt > 0:
                ratio = cnt * 1.0 / len(self.deg)
                info = f"num_core ['{size * (iters - 1)}' '{size * iters}']\t num: {cnt}\t ratio: {ratio}"
                info_str += info + "\n"
                if verbose:
                    print(info)
            else:
                break
            iters += 1

        max_deg = np.max(self.deg)
        cnt = len(np.where(self.deg == max_deg[0]))
        info = f"max-core: {max_deg}\t num: {cnt}\t ratio: {cnt / len(self.deg)}"
        info_str += info + "\n"
        if verbose:
            print(info)

        return info_str
