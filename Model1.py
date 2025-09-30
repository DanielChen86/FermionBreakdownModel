from itertools import chain, combinations, product

import numpy as np

from ModelBase import ModelBase
from TwoLayer import (Hamiltonian, HamiltonianBase, TwoLayerBasis)
from EntropyGallery import Entropy1

import matplotlib.pyplot as plt

import time


class Model1(ModelBase, Entropy1):
    def __init__(self, M: int, N: int,
                 mu0: float, J0: float, W0: float, Q: int = None,
                 random_J: bool = False, random_W: bool = True,
                 printer: bool = True):
        ModelBase.__init__(self)
        Entropy1.__init__(self)
        self.M = M
        self.N = N
        self.mu0 = mu0
        self.J0 = J0
        self.W0 = W0
        if Q:
            self.Q = Q
        else:
            self.Q = 3**(self.M-1)
        self.random_J = random_J
        self.random_W = random_W
        self.printer = printer

        self.gen_basis()

    def index(self, site: int, mode: int):
        return site * (self.N+1) + mode

    def index_inv(self, x: int):
        return x // (self.N+1), x % (self.N+1)

    def gen_coeffi(self):
        self.mu = {}
        for m in range(1, self.M+1):
            self.mu[m] = self.mu0

        self.J = {}
        for m in range(1, self.M):
            for i, j, k, l in product(*([range(1, self.N+1)]*4)):
                if (i < j) and (j < k):
                    if self.random_J:
                        self.J[m, i, j, k, l] = np.random.randn() * self.J0
                    else:
                        self.J[m, i, j, k, l] = self.J0

        self.v = {}
        for m in range(1, self.M+1):
            for i in range(1, self.N+1):
                if self.random_W:
                    self.v[m, i] = np.random.randn() * self.W0
                else:
                    self.v[m, i] = self.W0

    def Qfunc(self, num: list):
        num_arr = np.array(num)
        return np.sum(np.power(3, self.M-np.arange(self.M)-1) * num_arr)

    def gen_basis(self):
        self.electron_num = []
        for num in product(*([range(0, self.N+1)]*self.M)):
            if self.Qfunc(num) == self.Q:
                self.electron_num.append(num)

        self.basis_op: list[TwoLayerBasis] = []
        for basis in self.electron_num:
            iter = [list(combinations([i for i in range(1, self.N+1)], basis[m]))
                    for m in range(len(basis))]
            for mode in product(*(iter)):
                mode_fermion = [[(i, v_) for v_ in v]
                                for i, v in enumerate(mode)]
                mode_fermion = [x for x in mode_fermion if len(x) > 0]
                mode_fermion = list(chain.from_iterable(mode_fermion))
                mode_fermion = [self.index(x[0]+1, x[1]) for x in mode_fermion]
                self.basis_op.append(TwoLayerBasis(mode_fermion))
        self.len_basis = len(self.basis_op)
        if self.printer:
            print(f'len_basis = {self.len_basis}')

        self.basis_to_int = {}
        self.int_to_basis = {}
        for i, b in enumerate(self.basis_op):
            self.basis_to_int[b.__str__()] = i
            self.int_to_basis[i] = b.__str__()

    def gen_Hamiltonian(self):
        Hmu_init: list[HamiltonianBase] = []
        for m in range(1, self.M+1):
            for n in range(1, self.N+1):
                Hmu_init.append(HamiltonianBase(
                    self.mu[m], [self.index(m, n)], [self.index(m, n)]))
        self.Hmu = Hamiltonian(Hmu_init)

        HI_init: list[HamiltonianBase] = []
        for m in range(1, self.M):
            for i, j, k, l in product(*([range(1, self.N+1)]*4)):
                if (i < j) and (j < k):
                    hamiltonian_base = HamiltonianBase(self.J[m, i, j, k, l], [self.index(
                        m+1, i), self.index(m+1, j), self.index(m+1, k)], [self.index(m, l)])
                    HI_init.append(hamiltonian_base)
                    HI_init.append(hamiltonian_base.hermitian_conjugated())
        self.HI = Hamiltonian(HI_init)

        Hdis_init: list[HamiltonianBase] = []
        for m in range(1, self.M+1):
            for i in range(1, self.N+1):
                Hdis_init.append(HamiltonianBase(
                    self.v[m, i], [self.index(m, i)], [self.index(m, i)]))
        self.Hdis = Hamiltonian(Hdis_init)

        self.hamiltonian = self.Hmu + self.HI + self.Hdis

    def run_num(self):
        if self.printer:
            print('Start: Number matrix...')
        t0 = time.time()

        num_init: list[HamiltonianBase] = []
        for m in range(1, self.M+1):
            for n in range(1, self.N+1):
                num_init.append(HamiltonianBase(
                    1, [self.index(m, n)], [self.index(m, n)]))
        self.numtot = Hamiltonian(num_init)
        self.numtotMat = self.genMat(self.numtot)

        self.num = {}
        self.numMat = {}
        for m in range(1, self.M+1):
            num_init = []
            for n in range(1, self.N+1):
                num_init.append(HamiltonianBase(
                    1, [self.index(m, n)], [self.index(m, n)]))
            self.num[m] = Hamiltonian(num_init)
            self.numMat[m] = self.genMat(self.num[m])

        if self.printer:
            if self.len_basis > 300:
                print(
                    f'Complete: Number matrix (time consumed: {time.time() - t0} sec(s)).')
            else:
                print(
                    f'Complete: Number matrix (time consumed: {time.time() - t0} sec(s)).')

    def run_hamiltonian(self):
        self.gen_coeffi()
        self.gen_Hamiltonian() 
        self.gen_HamiltonianMat()


if __name__ == '__main__':
    N = 3
    M = 4
    J = 1
    Q = int(N*(3**M-1)/8)
    print(Q)
    model = Model1(M=M, N=N, mu0=1, J0=J, W0=0, random_J=False, Q = Q)
    model.run_hamiltonian()
    model.run_eigen()
    model.run_entropy()

    energies = np.sort(model.eig_values)
    spacings = energies[1:] - energies[:-1]

    np.savez_compressed(
        f'BT_ED_data_N{N}_M{M}_J{J}.npz',
        energies=energies,
        eigenvalues=model.eig_values,
        entropy=model.entanglement_entropy
    )

    # np.savez_compressed(
    #     f'BT_ED_data_N{N}_M{M}_J{J}_GS.npz',
    #     energies=energies,
    #     eigenvalues=model.eig_values,
    #     entropy=model.entanglement_entropy
    # )

    # plt.scatter(np.arange(model.len_basis), np.sort(model.eig_values), s=3)
    # plt.show()

    # plt.hist(model.eig_values_sorted[1:] - model.eig_values_sorted[:-1], 80)
    # plt.xlim(0, 5)
    # plt.ylim(0, 200)
    # plt.show()

    # plt.scatter(model.eig_values, model.entanglement_entropy, s=3)
    # plt.axhline(y=np.log(2), linewidth=1, linestyle='dotted', color='purple')
    # plt.show()
