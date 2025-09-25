from TwoLayer import TwoLayerBasis
from utils import list_to_str, H
import numpy as np
from itertools import product
import time
import sys


class EntropyBase:
    def __init__(self):
        self.M: int = None
        self.basis_op: list[TwoLayerBasis] = None
        self.basis_to_int: dict[str, int] = None
        self.len_basis: int = None
        self.basis_A: list[str] = None
        self.basis_B: list[str] = None
        self.len_basis_A: int = None
        self.len_basis_B: int = None
        self.int_to_basis: dict[int, str] = None
        self.basis_op_to_AB: dict[str, tuple(str)] = None
        self.basis_A_to_int: dict[str, int] = None
        self.int_to_basis_A: dict[int, str] = None
        self.basis_B_to_int: dict[str, int] = None
        self.int_to_basis_B: dict[int, str] = None
        self.HamiltonianMat: np.array = None
        self.eig_values: np.array = None
        self.eig_vectors: np.array = None
        self.printer: bool = None

    def run_entropy(self):
        '''
        Generate:
        self.entanglement_entropy: list[complex]
        '''
        raise Exception('Uninitialized function: Entropy.run_entropy().')

    def gen_AB_basis(self):
        '''
        Generate:
        self.basis_op_to_AB: dict[str, tuple(str)]
        self.basis_A: list[str]
        self.basis_B: list[str]
        self.len_basis_A: int
        self.len_basis_B: int
        self.basis_A_to_int: dict[str, int]
        self.int_to_basis_A: dict[int, str]
        self.basis_B_to_int: dict[str, int]
        self.int_to_basis_B: dict[int, str]
        '''
        raise Exception('Uninitialized function: Entropy.gen_AB_basis().')

    def gen_rhoAB_to_rhoA_map(self):
        if self.printer:
            print('Start: generate rhoAB to rhoA map...')
        t0 = time.time()
        self.rho_AB_to_rhoA_map: dict[(int, int), list[tuple[int]]] = {}
        rho_AB_to_rhoA_map: dict[(int, int), list[tuple[int]]] = {}
        for row_A, col_A in product(*([range(self.len_basis_A)]*2)):
            rho_AB_to_rhoA_map[row_A, col_A]: list[tuple[int]] = []
        for row, col in product(*([range(self.len_basis)]*2)):
            row_str = self.int_to_basis[row]
            col_str = self.int_to_basis[col]
            row_A_str, row_B_str = self.basis_op_to_AB[row_str]
            col_A_str, col_B_str = self.basis_op_to_AB[col_str]
            row_A = self.basis_A_to_int[row_A_str]
            col_A = self.basis_A_to_int[col_A_str]
            if row_B_str == col_B_str:
                rho_AB_to_rhoA_map[row_A, col_A].append((row, col))
        for row_A, col_A in product(*([range(self.len_basis_A)]*2)):
            map_value = list(zip(*rho_AB_to_rhoA_map[row_A, col_A]))
            if len(map_value) == 2:
                self.rho_AB_to_rhoA_map[row_A, col_A] = map_value
            elif len(map_value) == 0:
                self.rho_AB_to_rhoA_map[row_A, col_A] = [[], []]
            else:
                raise Exception(
                    f'Invalid value for rho_AB_to_rhoA_map: key = {(row_A, col_A)}')
        if self.printer:
            print(
                f'Complete: generate rhoAB to rhoA map (time consumed: {time.time() - t0} sec(s)).')

    def get_rho(self, vec: np.array):
        if not len(vec) == self.len_basis:
            raise Exception(
                f'Invalid state vector dimension: {len(vec)}, should be {self.len_basis}')
        return np.outer(vec, np.conjugate(vec))

    def get_rho_A(self, vec: np.array):
        if not len(vec) == self.len_basis:
            raise Exception(
                f'Invalid state vector dimension: {len(vec)}, should be {self.len_basis}')
        rho = self.get_rho(vec)
        rho_A = np.zeros((self.len_basis_A, self.len_basis_A), dtype=complex)
        for row_A, col_A in product(*([range(self.len_basis_A)]*2)):
            rho_A[row_A, col_A] = np.sum(
                rho[self.rho_AB_to_rhoA_map[row_A, col_A][0], self.rho_AB_to_rhoA_map[row_A, col_A][1]])
        return rho_A

    def run_entropy_with_AB_basis(self):
        if self.printer:
            print('Start: calculate entanglement entropy...')
        t0 = time.time()
        self.entanglement_entropy = []
        segment = self.len_basis // 20
        for i in range(self.len_basis):
            if self.printer:
                if self.len_basis > 300:
                    if (i+1) % segment == 0:
                        stage = i // segment + 1
                        sys.stdout.write('\r{0}'.format(
                            '|' + '-' * stage + '.' * (20-stage) + f'| {5 * stage}%'))
                        sys.stdout.flush()
            vec = self.eig_vectors[i]
            rho_A = self.get_rho_A(vec)
            entanglement = self.get_entanglement_entropy(rho_A)
            if np.isclose(np.imag(entanglement), 0):
                entanglement = np.real(entanglement)
            else:
                raise Exception(
                    f'Complex entanglement entropy: {entanglement}')
            self.entanglement_entropy.append(entanglement)
        if self.printer:
            if self.len_basis > 300:
                print(
                    f'\nComplete: calculate entanglement entropy (time consumed: {time.time() - t0} sec(s)).')
            else:
                print(
                    f'Complete: calculate entanglement entropy (time consumed: {time.time() - t0} sec(s)).')

    def get_entanglement_entropy(self, rho_A: np.array):
        eig_values_A = np.round(np.linalg.eig(rho_A)[0], 5)
        entanglement = -np.sum(eig_values_A * np.log(eig_values_A + 1e-6))
        if not np.isclose(np.imag(entanglement), 0):
            raise Exception('Complex entanglement entropy.')
        return np.real(entanglement)


class Entropy1(EntropyBase):
    def __init__(self):
        EntropyBase.__init__(self)

    def index(self, site: int, mode: int):
        pass

    def index_inv(self, x: int):
        pass

    def gen_AB_basis(self):
        self.basis_A = []
        self.basis_B = []
        self.basis_op_to_AB = {}
        for op in self.basis_op:
            state = [self.index_inv(x) for x in op.fermion_state_0]
            AB_dict = dict(zip(state, [x[0] <= self.M/2 for x in state]))
            basis_A = []
            basis_B = []
            for k, v in AB_dict.items():
                if v:
                    index_A = self.index(*k)
                    basis_A.append(index_A)
                else:
                    index_B = self.index(*k)
                    basis_B.append(index_B)
            basis_A = sorted(basis_A)[::-1]
            basis_B = sorted(basis_B)[::-1]
            basis_A_str = list_to_str(basis_A)
            basis_B_str = list_to_str(basis_B)
            self.basis_A.append(basis_A_str)
            self.basis_B.append(basis_B_str)
            self.basis_op_to_AB[op.__str__()] = (basis_A_str, basis_B_str)
        self.basis_A = list(set(self.basis_A))
        self.basis_B = list(set(self.basis_B))
        self.basis_A_to_int = {b: i for i, b in enumerate(self.basis_A)}
        self.int_to_basis_A = {i: b for i, b in enumerate(self.basis_A)}
        self.basis_B_to_int = {b: i for i, b in enumerate(self.basis_B)}
        self.int_to_basis_B = {i: b for i, b in enumerate(self.basis_B)}
        self.len_basis_A = len(self.basis_A)
        self.len_basis_B = len(self.basis_B)

    def run_entropy(self):
        self.gen_AB_basis()
        self.gen_rhoAB_to_rhoA_map()
        self.run_entropy_with_AB_basis()


class Entropy2(EntropyBase):
    def __init__(self):
        EntropyBase.__init__(self)

    def run_entropy(self, split: int = None):
        if not split:
            split = self.M // 2

        self.gen_AB_basis(split)
        self.gen_rhoAB_to_rhoA_map()
        self.run_entropy_with_AB_basis()

    def gen_AB_basis(self, split: int = None):
        if not split:
            split = self.M // 2

        self.basis_A = []
        self.basis_B = []
        self.basis_op_to_AB = {}
        for op in self.basis_op:
            fermion_state_arr = np.array(op.fermion_state_0)
            screen = fermion_state_arr > split
            basis_A_f_list = fermion_state_arr[np.logical_not(screen)].tolist()
            basis_B_f_list = fermion_state_arr[screen].tolist()
            basis_A_f_str = list_to_str(basis_A_f_list)
            basis_B_f_str = list_to_str(basis_B_f_list)
            basis_A_p_str = list_to_str(op.pauli_state[:split])
            basis_B_p_str = list_to_str(op.pauli_state[split:])
            basis_A_str = f'{basis_A_f_str};{basis_A_p_str}'
            basis_B_str = f'{basis_B_f_str};{basis_B_p_str}'
            self.basis_A.append(basis_A_str)
            self.basis_B.append(basis_B_str)
            self.basis_op_to_AB[op.__str__()] = (basis_A_str, basis_B_str)
        self.basis_A = list(set(self.basis_A))
        self.basis_B = list(set(self.basis_B))
        self.basis_A_to_int = {b: i for i, b in enumerate(self.basis_A)}
        self.int_to_basis_A = {i: b for i, b in enumerate(self.basis_A)}
        self.basis_B_to_int = {b: i for i, b in enumerate(self.basis_B)}
        self.int_to_basis_B = {i: b for i, b in enumerate(self.basis_B)}
        self.len_basis_A = len(self.basis_A)
        self.len_basis_B = len(self.basis_B)

    def gen_product_state(self):
        basis_A_information = {k: base_A.split(
            ';') for k, base_A in self.int_to_basis_A.items()}
        basis_B_information = {k: base_B.split(
            ';') for k, base_B in self.int_to_basis_B.items()}

        H2Q = H(2, self.Q)
        M_half = self.M / 2
        max_A = int(np.floor(M_half))
        max_B = int(np.ceil(M_half))
        H2Q = [x for x in H2Q if x[0] <= max_A and x[1] <= max_B]
        Q_dist = H2Q[np.random.randint(len(H2Q))]

        basis_A_subspace = [';'.join(v) for _, v in basis_A_information.items() if (
            (not v[0] == '') and len(v[0].split(',')) == Q_dist[0]) or (v[0] == '' and Q_dist[0] == 0)]
        basis_B_subspace = [';'.join(v) for _, v in basis_B_information.items() if (
            (not v[0] == '') and len(v[0].split(',')) == Q_dist[1]) or (v[0] == '' and Q_dist[1] == 0)]
        coeffi_A = {x: np.random.randn() + 1j * np.random.randn()
                    for x in basis_A_subspace}
        coeffi_B = {x: np.random.randn() + 1j * np.random.randn()
                    for x in basis_B_subspace}

        vec = np.zeros(self.len_basis, dtype=complex)
        AB_to_basis_op = {v: k for k, v in self.basis_op_to_AB.items()}
        for base_A, base_B in product(basis_A_subspace, basis_B_subspace):
            basis_str = AB_to_basis_op[base_A, base_B]
            idx = self.basis_to_int[basis_str]
            vec[idx] = coeffi_A[base_A] * coeffi_B[base_B]
        vec = vec / np.sqrt(np.sum(np.conjugate(vec) @ vec))
        return vec

    def gen_max_ee(self, sample=50):
        all_max_EE = []
        for _ in range(sample):
            vec = np.random.randn(self.len_basis) + 1j * \
                np.random.randn(self.len_basis)
            vec = vec / np.sqrt(np.conjugate(np.transpose(vec)) @ vec)
            rho = self.get_rho_A(vec)
            all_max_EE.append(self.get_entanglement_entropy(rho))
        self.max_ee = np.mean(np.array(all_max_EE))
