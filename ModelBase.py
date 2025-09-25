from TwoLayer import EfficientMatrix, Hamiltonian
import numpy as np
import pandas as pd
import scipy.sparse
import time
from math import factorial
import sys


class ModelBase(EfficientMatrix):
    def __init__(self):
        EfficientMatrix.__init__(self)

        self.len_basis: int = None
        self.hamiltonian: Hamiltonian = None
        self.HamiltonianMat: np.array = None
        self.eig_values: np.array = None
        self.eig_vectors: np.array = None
        self.eig_values_sorted: np.array = None
        self.printer: bool = None

    def gen_HamiltonianMat(self, sparse=False):
        if self.printer:
            print('Start: Hamiltonian matrix...')
        t0 = time.time()

        self.HamiltonianMat = self.genMat(self.hamiltonian, sparse=sparse)

        if self.printer:
            if self.len_basis > 300:
                print(
                    f'\nComplete: Hamiltonian matrix (time consumed: {time.time() - t0} sec(s)).')
            else:
                print(
                    f'Complete: Hamiltonian matrix (time consumed: {time.time() - t0} sec(s)).')

    def save_hamiltonian(self, real_path, imag_path=None, sep=','):
        if self.printer:
            print('Start: save Hamiltonian...')
        t0 = time.time()
        not_zero_indices = np.where(np.logical_not(
            np.isclose(self.HamiltonianMat, 0)))

        output_real, output_imag = [], []
        for i in range(len(not_zero_indices[0])):
            row = not_zero_indices[0][i]
            col = not_zero_indices[1][i]
            append_real = [row, col, np.real(self.HamiltonianMat[row, col])]
            append_imag = [row, col, np.imag(self.HamiltonianMat[row, col])]
            output_real.append(append_real)
            output_imag.append(append_imag)
        df = pd.DataFrame(output_real)
        df.to_csv(real_path, index=None, header=None, sep=sep)
        if not imag_path == None:
            df = pd.DataFrame(output_imag)
            df.to_csv(imag_path, index=None, header=None, sep=sep)

        if self.printer:
            print(
                f'Complete: save Hamiltonian (time consumed: {time.time() - t0} sec(s)).')

    def run_eigen(self, orthonormal=False):
        if self.printer:
            print('Start: solve eigen equation...')
        t0 = time.time()
        if not orthonormal:
            eig_values, eig_vectors = np.linalg.eig(self.HamiltonianMat)
        else:
            eig_values, eig_vectors = np.linalg.eigh(self.HamiltonianMat)
        if not np.all(np.isclose(np.imag(eig_values), 0)):
            raise Exception('Imaginary energy eigenvalues.')
        self.eig_values = np.real(eig_values)
        self.eig_vectors = np.transpose(eig_vectors)
        self.eig_values_sorted = np.sort(self.eig_values)
        if self.printer:
            print(
                f'Complete: solve eigen equation (time consumed: {time.time() - t0} sec(s)).')

    def gen_r_value(self):
        diff0 = self.eig_values_sorted[1:] - self.eig_values_sorted[:-1]
        diff = diff0[np.where(np.logical_not(np.isclose(diff0, 0)))[0]]
        if len(diff) < len(diff0):
            print(
                f'Warning: Discard zero eigenvalue diff ({len(diff0)-len(diff)}/{len(diff0)}).')
        diff1 = diff[:-1]
        diff2 = diff[1:]
        ratio12 = diff1 / diff2
        ratio21 = diff2 / diff1
        r_values = [min(ratio12[i], ratio21[i]) for i in range(len(diff)-1)]
        self.r_value = np.mean(r_values)
        if self.printer:
            print(f'r value = {self.r_value}')

    def Hexp(self, coeffi):
        return np.transpose(self.eig_vectors) @ np.diag(np.exp(coeffi * self.eig_values)) @ np.conjugate(self.eig_vectors)

    def Hevolve(self, time: float):
        return self.Hexp(-1j * time)

    def HevolveSparse(self, dt: float, order: int = 20):
        t0 = time.time()
        identity = scipy.sparse.eye(self.len_basis)
        HamiltonianMatSparse_pow = scipy.sparse.csc_matrix(
            identity, dtype=complex)
        returned = scipy.sparse.csc_matrix(identity, dtype=complex)
        print(f'Start calculate HevolveSparse...')
        for n in range(1, order+1):
            HamiltonianMatSparse_pow = HamiltonianMatSparse_pow @ self.HamiltonianMat
            returned += ((-1j * dt)**n / factorial(n)) * \
                HamiltonianMatSparse_pow
            sys.stdout.write('\r{0}'.format(
                '|' + '-' * (n) + '.' * (order-n)) + '|')
            sys.stdout.flush()
        print()
        print(
            f'Complete calculate HevolveSparse (time consumed: {time.time() - t0} sec(s)).')
        return returned
