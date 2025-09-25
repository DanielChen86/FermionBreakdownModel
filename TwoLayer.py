from copy import deepcopy

import numpy as np
import scipy.sparse
from openfermion.ops import FermionOperator

from utils import normal_vacuum

import sys
from typing import List, Dict


class FermionState:
    def __init__(self, fermion_state: List[int]):
        self.fermion_state_0 = sorted(fermion_state)[::-1]
        self.fermion_state = FermionOperator(
            [(i, 1) for i in self.fermion_state_0])


def PauliOperateState(pauli: int, state: int):
    if pauli == 0:
        return 1, state
    elif pauli == 1:
        return 1, int(state == 0)
    elif pauli == 2:
        return -2j * int(state == 0) + 1j, int(state == 0)
    elif pauli == 3:
        return -2 * int(state == 0) + 1, state
    elif pauli == 4:    # raising operator
        return int(state == 0), 1
    elif pauli == 5:    # lowering operator
        return int(state == 1), 0
    elif pauli == 6:    # number operator
        return int(state == 1), state
    else:
        raise Exception(f'Incorrect Pauli matrix: pauli = {pauli}')


class PauliState:
    def __init__(self, pauli_state: List[int]):
        self.pauli_state = pauli_state


class PauliOperator:
    def __init__(self, op: Dict[int, int]):
        self.op = op

    def act(self, pauli_state: PauliState):
        coeffi_all = 1
        new_state = pauli_state.pauli_state
        for k, v in self.op.items():
            coeffi, state = PauliOperateState(v, new_state[k])
            if np.isclose(coeffi, 0):
                return 0, None
            new_state[k] = state
            coeffi_all = coeffi_all * coeffi
        return coeffi_all, new_state


class TwoLayerBasis(FermionState, PauliState):
    def __init__(self, fermion_state: List[int], pauli_state: List[int] = [0]):
        FermionState.__init__(self, fermion_state)
        PauliState.__init__(self, pauli_state)

    def __repr__(self) -> str:
        return ','.join(map(str, self.fermion_state_0)) + ';' + ','.join(map(str, self.pauli_state))

    def __str__(self) -> str:
        return self.__repr__()


class HamiltonianBase:
    def __init__(self, coeffi: complex, fermion_creation_op: List[int] = [], fermion_annihilation_op: List[int] = [], pauli_op: Dict[int, int] = {}) -> None:
        self.fermion_creation_op_0 = fermion_creation_op
        self.fermion_annihilation_op_0 = fermion_annihilation_op
        self.pauli_op_0 = {k-1: v for k, v in pauli_op.items()}
        self.fermion_op = FermionOperator(tuple(
            [(i, 1) for i in fermion_creation_op])) * FermionOperator(tuple([(i, 0) for i in fermion_annihilation_op]))
        self.pauli_op = PauliOperator(self.pauli_op_0)
        self.coeffi = coeffi

    def hermitian_conjugated(self):
        new_fermion_creation_op = self.fermion_annihilation_op_0[::-1]
        new_fermion_annihilation_op = self.fermion_creation_op_0[::-1]
        new_pauli_op = self.pauli_op_0
        for k, _ in new_pauli_op:
            if k == 4:
                new_pauli_op[k] = 5
            elif k == 5:
                new_pauli_op[k] = 4
        return HamiltonianBase(np.conj(self.coeffi), new_fermion_creation_op, new_fermion_annihilation_op, new_pauli_op)

    def O_ket(self, ket0: TwoLayerBasis):
        ket = deepcopy(ket0)
        coeffi = self.coeffi
        new_fermion_0 = normal_vacuum(self.fermion_op * ket.fermion_state)
        if new_fermion_0 == None:
            return None, None
        for k, v in new_fermion_0.terms.items():
            if len(k) > 0:
                new_fermion_0 = [x[0] for x in k]
                coeffi = coeffi * v
            elif not np.isclose(v, 0):
                new_fermion_0 = []
                coeffi = coeffi * v
                # should be careful
                raise Exception('Warning')
        if np.isclose(coeffi, 0):
            return None, None
        pauli_coeffi, new_paulistate = self.pauli_op.act(ket)
        coeffi = coeffi * pauli_coeffi
        if np.isclose(coeffi, 0):
            return None, None
        return coeffi, TwoLayerBasis(new_fermion_0, new_paulistate)

    def __repr__(self) -> str:
        pauli_str = ','.join(
            [f'{k+1}:{v}' for k, v in self.pauli_op_0.items()])
        return f'{self.coeffi};' + ','.join(map(str, self.fermion_creation_op_0)) + ';' + ','.join(map(str, self.fermion_annihilation_op_0)) + ';' + pauli_str

    def __str__(self) -> str:
        return self.__repr__()


class Hamiltonian:
    def __init__(self, hamiltonians: List[HamiltonianBase]):
        self.hamiltonians = hamiltonians

    def __add__(self, __o: object):
        return Hamiltonian(self.hamiltonians + __o.hamiltonians)

    def __repr__(self) -> str:
        return '||\n'.join([hb.__str__() for hb in self.hamiltonians])

    def __str__(self) -> str:
        return self.__repr__()


class EfficientMatrix:
    def __init__(self):
        self.basis_op: List[TwoLayerBasis] = None
        self.len_basis: int = None
        self.basis_to_int: Dict[str, int] = None
        self.printer: bool = None

    def genMat(self, op: Hamiltonian, printer=None, sparse=False):
        if printer == None:
            printer = self.printer
        if not sparse:
            returnMat = np.zeros((self.len_basis, self.len_basis)) + \
                1j * np.zeros((self.len_basis, self.len_basis))
        else:
            returnMat = scipy.sparse.lil_matrix(
                (self.len_basis, self.len_basis), dtype=complex)

        segment = self.len_basis // 20
        for i in range(self.len_basis):
            b1 = self.basis_op[i]
            if printer:
                if self.len_basis > 300:
                    if (i+1) % segment == 0:
                        stage = i // segment + 1
                        sys.stdout.write('\r{0}'.format(
                            '|' + '-' * stage + '.' * (20-stage) + f'| {5 * stage}%'))
                        sys.stdout.flush()
            for h in op.hamiltonians:
                coeffi, o_ket = h.O_ket(b1)
                if not coeffi == None:
                    b2_str = o_ket.__str__()
                    try:
                        row = self.basis_to_int[b1.__str__()]
                        col = self.basis_to_int[b2_str]
                        returnMat[col, row] += coeffi
                    except:
                        print(b1, h)
                        raise Exception(
                            f'Incomplete basis: HermitianBase = ({h}), ket = ({b1.__str__()}), bra = ({b2_str}), coeffi = {coeffi}')
        if not sparse:
            if np.all(np.isclose(np.imag(returnMat), 0)):
                returnMat = np.real(returnMat)
        else:
            returnMat = scipy.sparse.csc_matrix(returnMat)
        return returnMat


if __name__ == '__main__':
    em = EfficientMatrix()
    b1 = TwoLayerBasis([2, 5, 6, 8])
    b2 = TwoLayerBasis([1, 2, 8])
    hb1 = HamiltonianBase(1, [1], [5, 6])
    h = Hamiltonian([hb1])
    em.basis_op = [b1, b2]
    em.basis_to_int = {}
    em.basis_to_int[b1.__str__()] = 0
    em.basis_to_int[b2.__str__()] = 1
    em.len_basis = 2
    em.printer = False

    print(em.genMat(h))
