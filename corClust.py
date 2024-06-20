import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, to_tree


class corClust:
    def __init__(self, n):
        self.n = n
        self.c = np.zeros(n)
        self.c_r = np.zeros(n)
        self.c_rs = np.zeros(n)
        self.C = np.zeros((n, n))
        self.N = 0

    def update(self, x):
        self.N += 1
        self.c += x
        c_rt = x - self.c / self.N
        self.c_r += c_rt
        self.c_rs += c_rt ** 2
        self.C += np.outer(c_rt, c_rt)

    def corrDist(self):
        c_rs_sqrt = np.sqrt(self.c_rs)
        C_rs_sqrt = np.outer(c_rs_sqrt, c_rs_sqrt)
        C_rs_sqrt[
            C_rs_sqrt == 0] = 1e-100
        D = 1 - self.C / C_rs_sqrt
        D[
            D < 0] = 0
        return D

    def cluster(self, maxClust):
        D = self.corrDist()
        Z = linkage(D[np.triu_indices(self.n, 1)])
        if maxClust < 1:
            maxClust = 1
        if maxClust > self.n:
            maxClust = self.n
        map = self.__breakClust__(to_tree(Z), maxClust)
        return map

    def __breakClust__(self, dendro, maxClust):
        if dendro.count <= maxClust:
            return [dendro.pre_order()]
        return self.__breakClust__(dendro.get_left(), maxClust) + self.__breakClust__(dendro.get_right(), maxClust)

