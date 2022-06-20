"""
ForceField class for calculating energy, gradient, hessian, and dynamical matrix.
This module internally uses a dynamic link library forcefield.dll compiled by Fortran.
"""

import math
import os
import sys
from ctypes import (POINTER, byref, c_char_p, c_double, c_int, c_void_p,
                    create_string_buffer)
from typing import List, Tuple

import numpy as np
from numpy import ndarray
from numpy.ctypeslib import ndpointer
from matsuilab import csp


class ForceField():
    """Force field consisting of multiple terms.
    ForceField object can be created from an ff file by a class method, ForceField.open("sample.ff").

    Attributes:
        na: the number of atoms per molecule (int)
        bonded_terms: list of boded force field terms (List[BondedTerm])
        non_bonded_terms: list of non-bonded force field terms (List[NonBondedTerm])
        exclusion: exclusion matrix used for excluding the non-bonded interactions between (super-)bonded atoms (ndarray(shape=(na, na), dtype=float))
    """

    def __init__(self, na:int=0):
        self.na : int = na
        self.bonded_terms : List[BondedTerm] = []
        self.non_bonded_terms : List[NonBondedTerm] = []
        self.ewald_terms : List[EwaldTerm] = []
        self.exclusion : ndarray = np.zeros((na, na))

    @classmethod
    def read(cls, data:str):
        self = cls()
        lines = data.splitlines()
        i = 0
        while i < len(lines):
            tokens = lines[i].lower().split()
            if len(tokens) == 0:
                pass
            elif tokens[0] == "num_of_atoms_per_mol":
                self.na = int(tokens[1])
            elif tokens[0] == "bonded":
                term = BondedTerm(tokens[1], self.na, int(tokens[2]))
                for j in range(term.ngrp):
                    i = i + 1
                    tokens = lines[i].lower().split()
                    term.idx[j, :] = [int(s) for s in tokens[:term.nidx]]
                    term.par[j, :] = [float(s) for s in tokens[term.nidx:]]
                self.bonded_terms.append(term)
            elif tokens[0] == "non_bonded":
                term = NonBondedTerm(tokens[1], self.na)
                for j in range(term.na):
                    for k in range(term.na):
                        i = i + 1
                        tokens = lines[i].split()
                        term.par[j, k, :] = [float(s) for s in tokens]
                term.exclusion = self.exclusion
                self.non_bonded_terms.append(term)
            elif tokens[0] == "ewald":
                term = EwaldTerm(tokens[1], self.na)
                for j in range(term.na):
                    i = i + 1
                    tokens = lines[i].split()
                    term.par[j, :] = [float(s) for s in tokens]
                term.exclusion = self.exclusion
                self.ewald_terms.append(term)
            elif tokens[0] == "exclusion":
                self.exclusion = np.zeros((self.na, self.na))
                for j in range(self.na):
                    i = i + 1
                    tokens = lines[i].split()
                    self.exclusion[j, :] = [float(s) for s in tokens]
                for term in self.non_bonded_terms:
                    term.exclusion = self.exclusion
                for term in self.ewald_terms:
                    term.exclusion = self.exclusion
            else:
                raise Exception("Unknown keyword: " + tokens[0])
            i = i + 1
        return self

    @classmethod
    def open(cls, file:str):
        with open(file, "r") as f:
            return cls.read(f.read())

    # def fun(self):
    #     return lambda x: self.energy(x[:-9], x[-9:])
    
    # def gfun2(self, n, delta=1e-4):

    #     gradients = np.zeros((n+3, 3))
    #     grad1 = gradients[:-3, :] # view
    #     grad2 = gradients[-3:, :] # view

    #     def g(x):
    #         self.calc_gradients(x[:-9], x[-9:], grad1, grad2, delta)
    #         return np.reshape(gradients, gradients.size)

    #     return g

    # def hfun1(self, n, delta=1e-4):
        
    #     hess11 = np.zeros((n, 3, n, 3))

    #     def h(x):
    #         self.calc_hessian(x[:-9], x[-9:], hess11, delta)
    #         return np.reshape(hess11, (n*3, n*3))

    #     return h

    # def hfun2(self, n, delta=1e-4):

    #     hessian = np.zeros((n+3, 3, n+3, 3))
    #     hess11 = np.zeros((n, 3, n, 3))
    #     hess12 = np.zeros((n, 3, 3, 3))
    #     hess22 = np.zeros((3, 3, 3, 3))

    #     def h(x):
    #         self.calc_hessians(x[:-9], x[-9:], hess11, hess12, hess22, delta)
    #         hessian[:-3, :, :-3, :] = hess11
    #         hessian[:-3, :, -3:, :] = hess12
    #         hessian[-3:, :, :-3, :] = hess12.transpose((2, 3, 0, 1))
    #         hessian[-3:, :, -3:, :] = hess22
    #         return np.reshape(hessian, (x.size, x.size))

    #     return h

    # def rfun(self, interval:int=10):
        def r(o:csp.DictLike):
            if interval > 0 and o.cycle % interval == 0:
                self.refresh(o.x[:-9], o.x[-9:])
                return True
            else:
                return False
        return r

    def refresh(self, atoms:ndarray, lattice:ndarray):
        atoms = np.reshape(atoms, (atoms.size // 3, 3))
        lattice = np.reshape(lattice, (3, 3))
        for t in self.non_bonded_terms:
            t.refresh(atoms, lattice)
        for t in self.ewald_terms:
            t.refresh(atoms, lattice)

    def energy(self, atoms:ndarray, lattice:ndarray=None) -> float:
        atoms = np.reshape(atoms, (atoms.size // 3, 3))
        lattice = np.reshape(lattice, (3, 3))
        e = 0.0
        for t in self.ewald_terms:
            e += t.energy(atoms, lattice)
        for t in self.non_bonded_terms:
            e += t.energy(atoms, lattice)
        for t in self.bonded_terms:
            e += t.energy(atoms)
        return e

    def calc_gradients(self, atoms:ndarray, lattice:ndarray,
                       grad1:ndarray, grad2:ndarray,
                       delta:float=1e-4, reset:bool=True):
        atoms = np.reshape(atoms, (atoms.size // 3, 3))
        lattice = np.reshape(lattice, (3, 3))
        grad1 = np.reshape(grad1, (grad1.size // 3, 3))
        grad2 = np.reshape(grad2, (3, 3))
        if reset:
            grad1.fill(0.0)
            grad2.fill(0.0)
        for t in self.ewald_terms:
            t.add_gradient(atoms, lattice, grad1, grad2, delta)
        for t in self.non_bonded_terms:
            t.add_gradient(atoms, lattice, grad1, grad2, delta)
        for t in self.bonded_terms:
            t.add_gradient(atoms, grad1, delta)
        return grad1, grad2

    def calc_hessian(self, atoms:ndarray, lattice:ndarray,
                     hess11:ndarray,
                     delta:float=1e-4, reset:bool=True):
        atoms = np.reshape(atoms, (atoms.size // 3, 3))
        lattice = np.reshape(lattice, (3, 3))
        hess11 = np.reshape(hess11, (atoms.size // 3, 3, atoms.size // 3, 3))
        if reset:
            hess11.fill(0.0)
        for t in self.ewald_terms:
            t.add_hessian(atoms, lattice, hess11, delta)
        for t in self.non_bonded_terms:
            t.add_hessian(atoms, lattice, hess11, delta)
        for t in self.bonded_terms:
            t.add_hessian(atoms, hess11, delta)

    def calc_hessians(self, atoms:ndarray, lattice:ndarray,
                      hess11:ndarray, hess12:ndarray, hess22:ndarray,
                      delta:float=1e-4, reset:bool=True):
        atoms = np.reshape(atoms, (atoms.size // 3, 3))
        lattice = np.reshape(lattice, (3, 3))
        hess11 = np.reshape(hess11, (atoms.size // 3, 3, atoms.size // 3, 3))
        hess12 = np.reshape(hess12, (atoms.size // 3, 3, 3, 3))
        hess22 = np.reshape(hess22, (3, 3, 3, 3))
        if reset:
            hess11.fill(0.0)
            hess12.fill(0.0)
            hess22.fill(0.0)
        for t in self.ewald_terms:
            t.add_hessians(atoms, lattice, hess11, hess12, hess22, delta)
        for t in self.non_bonded_terms:
            t.add_hessians(atoms, lattice, hess11, hess12, hess22, delta)
        for t in self.bonded_terms:
            t.add_hessian(atoms, hess11, delta)

    def calc_dynamical(self, atoms:ndarray, lattice:ndarray,
                       qv:ndarray, dyna:ndarray,
                       delta:float=1e-4, reset:bool=True):
        atoms = np.reshape(atoms, (atoms.size // 3, 3))
        lattice = np.reshape(lattice, (3, 3))
        dyna = np.reshape(dyna, (atoms.size // 3, 3, atoms.size // 3, 3))
        if reset:
            dyna.fill(0,0)
        for t in self.ewald_terms:
            t.add_dynamical(atoms, lattice, qv, dyna, delta)
        for t in self.non_bonded_terms:
            t.add_dynamical(atoms, lattice, qv, dyna, delta)
        for t in self.bonded_terms:
            t.add_dynamical(atoms, dyna, delta)
        return dyna

    def calc_numerical_gradients(self, atoms:ndarray, lattice:ndarray,
                                 grad1:ndarray, grad2:ndarray,
                                 delta:float=1e-4):

        atoms = np.reshape(atoms, (atoms.size // 3, 3))
        lattice = np.reshape(lattice, (3, 3))
        grad1 = np.reshape(grad1, (grad1.size // 3, 3))
        grad2 = np.reshape(grad2, (3, 3))

        # numerical differentiation by atom coordinates
        for i in range(atoms.shape[0]):
            for j in range(3):
                old = atoms[i, j]

                atoms[i, j] = old - delta
                e1 = self.energy(atoms, lattice)

                atoms[i, j] = old + delta
                e2 = self.energy(atoms, lattice)

                atoms[i, j] = old
                grad1[i, j] = (e2 - e1) / (2 * delta)

        # numerical differentiation by lattice coordinates
        for i in range(3):
            for j in range(3):
                old = lattice[i, j]

                lattice[i, j] = old - delta
                e1 = self.energy(atoms, lattice)

                lattice[i, j] = old + delta
                e2 = self.energy(atoms, lattice)

                lattice[i, j] = old
                grad2[i, j] = (e2 - e1) / (2 * delta)

    def calc_numerical_hessian(self, atoms:ndarray, lattice:ndarray,
                               hess11:ndarray, delta:float=1e-4,
                               indices:Tuple[int]=None):
        if indices is None:
            indices = range(atoms.size // 3)
        n = len(indices)

        atoms = np.reshape(atoms, (atoms.size // 3, 3))
        lattice = np.reshape(lattice, (3, 3))
        hess11 = np.reshape(hess11, (n, 3, n, 3))

        for i in range(n):
            for s in range(3):
                for j in range(n):
                    for t in range(3):
                        p = indices[i]
                        q = indices[j]
                        old1 = atoms[p][s]
                        old2 = atoms[q][t]
                        atoms[p][s] -= delta
                        atoms[q][t] -= delta
                        e11 = self.energy(atoms, lattice)
                        atoms[p][s] = old1
                        atoms[q][t] = old2
                        atoms[p][s] -= delta
                        atoms[q][t] += delta
                        e12 = self.energy(atoms, lattice)
                        atoms[p][s] = old1
                        atoms[q][t] = old2
                        atoms[p][s] += delta
                        atoms[q][t] -= delta
                        e21 = self.energy(atoms, lattice)
                        atoms[p][s] = old1
                        atoms[q][t] = old2
                        atoms[p][s] += delta
                        atoms[q][t] += delta
                        e22 = self.energy(atoms, lattice)
                        atoms[p][s] = old1
                        atoms[q][t] = old2
                        hess11[i, s, j, t] = (e22 + e11 - e12 - e21) / (2 * delta)**2

    def calc_numerical_hess12(self, atoms:ndarray, lattice:ndarray,
                              hess12:ndarray, delta:float=1e-4,
                              indices:Tuple[int]=None):
        if indices is None:
            indices = range(atoms.size // 3)
        n = len(indices)

        atoms = np.reshape(atoms, (atoms.size // 3, 3))
        lattice = np.reshape(lattice, (3, 3))
        hess12 = np.reshape(hess12, (n, 3, 3, 3))

        for i in range(n):
            for s in range(3):
                for j in range(3):
                    for t in range(3):
                        p = indices[i]
                        old1 = atoms[p][s]
                        old2 = lattice[j][t]
                        atoms[p][s] -= delta
                        lattice[j][t] -= delta
                        e11 = self.energy(atoms, lattice)
                        atoms[p][s] = old1
                        lattice[j][t] = old2
                        atoms[p][s] -= delta
                        lattice[j][t] += delta
                        e12 = self.energy(atoms, lattice)
                        atoms[p][s] = old1
                        lattice[j][t] = old2
                        atoms[p][s] += delta
                        lattice[j][t] -= delta
                        e21 = self.energy(atoms, lattice)
                        atoms[p][s] = old1
                        lattice[j][t] = old2
                        atoms[p][s] += delta
                        lattice[j][t] += delta
                        e22 = self.energy(atoms, lattice)
                        atoms[p][s] = old1
                        lattice[j][t] = old2
                        hess12[i, s, j, t] = (e22 + e11 - e12 - e21) / (2 * delta)**2

    def calc_numerical_hess22(self, atoms:ndarray, lattice:ndarray,
                              hess22:ndarray, delta:float=1e-4):

        atoms = np.reshape(atoms, (atoms.size // 3, 3))
        lattice = np.reshape(lattice, (3, 3))
        hess22 = np.reshape(hess22, (3, 3, 3, 3))

        for i in range(3):
            for s in range(3):
                for j in range(3):
                    for t in range(3):
                        old1 = lattice[i][s]
                        old2 = lattice[j][t]
                        lattice[i][s] -= delta
                        lattice[j][t] -= delta
                        e11 = self.energy(atoms, lattice)
                        lattice[i][s] = old1
                        lattice[j][t] = old2
                        lattice[i][s] -= delta
                        lattice[j][t] += delta
                        e12 = self.energy(atoms, lattice)
                        lattice[i][s] = old1
                        lattice[j][t] = old2
                        lattice[i][s] += delta
                        lattice[j][t] -= delta
                        e21 = self.energy(atoms, lattice)
                        lattice[i][s] = old1
                        lattice[j][t] = old2
                        lattice[i][s] += delta
                        lattice[j][t] += delta
                        e22 = self.energy(atoms, lattice)
                        lattice[i][s] = old1
                        lattice[j][t] = old2
                        hess22[i, s, j, t] = (e22 + e11 - e12 - e21) / (2 * delta)**2

        return hess22

    def check_numerical_gradients(self, atoms:ndarray, lattice:ndarray, delta:float=1e-4):

        atoms = np.reshape(atoms, (atoms.size // 3, 3))
        lattice = np.reshape(lattice, (3, 3))
        n = atoms.shape[0]

        def summary(error):
            if error < 1e-6:
                return 'Good'
            elif error < 1e-4:
                return 'Fair'
            else:
                return 'Bad'

        print()
        print('==== Check Numerical Gradients ====')

        num_grad1 = np.zeros((n, 3))
        num_grad2 = np.zeros((3, 3))
        self.calc_numerical_gradients(atoms, lattice, num_grad1, num_grad2, delta)

        grad1 = np.zeros((n, 3))
        grad2 = np.zeros((3, 3))
        self.calc_gradients(atoms, lattice, grad1, grad2, delta)

        dif1 = grad1 - num_grad1
        dif2 = grad2 - num_grad2

        print()
        print('-- Gradient with Atom Coordinates --')
        print()
        print('Analytical                       Numerical                        Difference')
        for i in range(atoms.shape[0]):
            values = list(grad1[i, :]) + list(num_grad1[i, :]) + list(dif1[i, :])
            print(' '.join([format(x, ' 6.3e') for x in values]))
        print()
        error = np.sqrt(np.mean(np.square(dif1))) / np.sqrt(np.mean(np.square(num_grad1)))
        print(f'RMS Dif./RMS Num.:  {error:6.3e}  {summary(error)}')
        error = np.max(abs(dif1))/np.max(abs(num_grad1))
        print(f'Max Dif./Max Num.:  {error:6.3e}  {summary(error)}')

        print()
        print('-- Gradient with Lattice Coordinates --')
        print()
        print('Analytical                       Numerical                        Difference')
        for i in range(3):
            values = list(grad2[i, :]) + list(num_grad2[i, :]) + list(dif2[i, :])
            print(' '.join([format(x, ' 6.3e') for x in values]))
        print()
        error = np.sqrt(np.mean(np.square(dif2))) / np.sqrt(np.mean(np.square(num_grad2)))
        print(f'RMS Dif./RMS Num.:  {error:6.3e}  {summary(error)}')
        error = np.max(abs(dif2))/np.max(abs(num_grad2))
        print(f'Max Dif./Max Num.:  {error:6.3e}  {summary(error)}')

    def check_numerical_hessians(self, atoms, lattice, delta=1e-4, indices='auto'):

        def summary(error):
            if error < 1e-6:
                return 'Good'
            elif error < 1e-4:
                return 'Fair'
            else:
                return 'Bad'

        print()
        print('==== Check Numerical Hessian ====')

        # calculate numerical hessians
        indices = (0, atoms.size // 3 // 2, atoms.size // 3 - 1)
        n = len(indices)
        num_hess11 = np.zeros((n, 3, n, 3))
        num_hess12 = np.zeros((n, 3, 3, 3))
        num_hess22 = np.zeros((3, 3, 3, 3))
        self.calc_numerical_hessian(atoms, lattice, num_hess11, delta, indices)
        self.calc_numerical_hess12(atoms, lattice, num_hess12, delta, indices)
        self.calc_numerical_hess22(atoms, lattice, num_hess22, delta)

        # calculate analytical hessians
        n = atoms.shape[0]
        hess11 = ndarray((n, 3, n, 3))
        hess12 = ndarray((n, 3, 3, 3))
        hess22 = ndarray((3, 3, 3, 3))
        self.calc_hessians(atoms, lattice, hess11, hess12, hess22, delta)

        # reshape
        hess11 = hess11[indices, :, :, :][:, :, indices, :]
        hess12 = hess12[indices, :, :, :]

        n = len(indices)

        print()
        print('-- Picked Up Atoms --')
        print()
        print(*indices)

        dif11 = hess11 - num_hess11
        dif12 = hess12 - num_hess12
        dif22 = hess22 - num_hess22

        # print()
        # print('-- Analytical --')
        # print()
        # for h in np.reshape(hess11, (n*3, n*3)):
        #     print(' '.join([format(x, ' 6.3e') for x in h]))
        # print()

        # print('-- Numerical --')
        # print()
        # for h in np.reshape(num_hess11, (n*3, n*3)):
        #     print(' '.join([format(x, ' 6.3e') for x in h]))
        # print()

        # print('-- Difference --')
        # print()
        # for h in np.reshape(dif11, (n*3, n*3)):
        #     print(' '.join([format(x, ' 6.3e') for x in h]))
        # print()

        print()
        print('-- Summary --')
        print()
        print('hess11')
        error = np.sqrt(np.mean(np.square(dif11))) / np.sqrt(np.mean(np.square(num_hess11)))
        print(f'RMS Dif./RMS Num.:  {error:6.3e}  {summary(error)}')
        error = np.max(abs(dif11))/np.max(abs(num_hess11))
        print(f'Max Dif./Max Num.:  {error:6.3e}  {summary(error)}')
        print()
        print('hess12')
        error = np.sqrt(np.mean(np.square(dif12))) / np.sqrt(np.mean(np.square(num_hess12)))
        print(f'RMS Dif./RMS Num.:  {error:6.3e}  {summary(error)}')
        error = np.max(abs(dif12))/np.max(abs(num_hess12))
        print(f'Max Dif./Max Num.:  {error:6.3e}  {summary(error)}')
        print()
        print('hess22')
        error = np.sqrt(np.mean(np.square(dif22))) / np.sqrt(np.mean(np.square(num_hess22)))
        print(f'RMS Dif./RMS Num.:  {error:6.3e}  {summary(error)}')
        error = np.max(abs(dif22))/np.max(abs(num_hess22))
        print(f'Max Dif./Max Num.:  {error:6.3e}  {summary(error)}')

    # def minimize_lattice_energy(self, atoms:ndarray, lattice:ndarray,
    #     opt_lattice:bool=True, newton:bool=True, conjugate:bool=False,
    #     maxcyc:int=1000, xtol:float=None, ytol:float=1e-9, gtol:float=None,
    #     alpha0:float=1.0, maxstep:float=1.0, linecyc:int=10, trajectory:str=None,
    #     report:Callable=None,
    #     refresh_interval:int=10, report_interval:int=1):
    #     """perform lattice energy minimization. atoms[0, :], lattice[0, 1], lattice[0, 2], lattice[1, 2] are unchanged.
        
    #     Args:
    #         ff: force field
    #         opt_lattice: whether lattice vectors are optimized
    #         newton: whether modified newton method is used
    #         conjugate: whether conjugate gradient method is used
    #         maxcyc: maximum number of cycles
    #         xtol: tolerance for the norm of delta-x
    #         ytol: tolerance for delta-y
    #         gtol: tolerance for the norm of gradient
    #         alpha0: initial guess of step in line search
    #         maxstep: upper limit of one step
    #         linecyc: maximum number of cycles for line search. If linecyc is 0 or None, the given alpha is used as is.
    #         trajectory: cif file name for trajectory
    #         refresh_interval: interval steps for refreshing
    #         report_interval: interval steps for printing report
    #         """

    #     n = atoms.size // 3
    #     atoms = np.reshape(atoms, atoms.size)
    #     lattice = np.reshape(lattice, lattice.size)
    #     x0 = np.concatenate(atoms, lattice) # copy

    #     if opt_lattice:
    #         mask = np.array(tuple([i >= 3 for i in range(n*3)]) + (True, False, False, True, True, False, True, True, True), dtype=bool)
    #     else:
    #         mask = np.array(tuple([i >= 3 for i in range(n*3)]) + ((False,)*9), dtype=bool)

    #     def fun(x):
    #         return self.energy(x[:-9], x[-9:])

    #     grad, grad1, grad2 = _empty_gradients(n)

    #     def jac(x):
    #         self.calc_gradients(x[:-9], x[-9:], grad1, grad2)
    #         return np.reshape(grad, (n+3)*3)

    #     if newton:
    #         hess, hess11, hess12, hess22 = _empty_hessians(n)

    #         def hfun(x):
    #             self.calc_hessians(x[:-9], x[-9:], hess11, hess12, hess22)
    #             return np.reshape(_join_hessians(hess11, hess12, hess22, hess), ((n+3)*3, (n+3)*3))
    #     else:
    #         hfun = None

    #     def refresh(o:csp.DictLike):
    #         if refresh_interval > 0 and o.cycle % refresh_interval == 0:
    #             self.refresh(o.x[:-9], o.x[-9:])
    #             return True
    #         else:
    #             return False

    #     result = csp.optimize(fun=fun, jac=jac, x0=x0, mask=mask,
    #         conjugate=conjugate, hess=hfun,
    #         maxcyc=maxcyc, xtol=xtol, ytol=ytol, gtol=gtol,
    #         alpha0=alpha0, maxstep=maxstep, linecyc=linecyc,
    #         refresh=refresh, report=report)

    #     atoms[:] = result.x[:-9]
    #     lattice[:] = result.x[-9:]

    #     return result


class BondedTerm():
    """Force field term of bonded interactions

    Attributes:
        name: the name of force field such as "stretch" (str)
        na: the number of atoms per molecule (int)
        ngrp: the number of interacting atom groups per molecule (int)
        nidx: the number of atoms per interacting atom group (int)
        npar: the number of force field parameters for each interacting atom group (int)
        idx: indices of interacting atoms (ndarray(shape=(ngrp, nidx), dtype=int))
        par: force field parameters such as spring constant (ndarray(shape=(ngrp, npar), dtype=float))
    """

    def __init__(self, name:str, na:int, ngrp:int):
        self.name = name
        self.na = na
        self.ngrp = ngrp
        if self.name == "stretch":
            self.nidx = 2
            self.npar = 2
        elif self.name == "bend" or self.name == "bend_cos":
            self.nidx = 3
            self.npar = 2
        elif self.name == "torsion":
            self.nidx = 4
            self.npar = 3
        elif self.name == "inversion" or self.name == "inversion_cos_planar":
            self.nidx = 4
            self.npar = 1
        #---- add new force field type here ----#
        else:
            raise NotImplementedError(f"Force field type not found: {name}")
        self.idx = np.zeros((ngrp, self.nidx), dtype=int)
        self.par = np.zeros((ngrp, self.npar), dtype=float)

    def energy(self, atoms:ndarray):
        """returns energy

        Args:
            atoms (ndarray): cartesian coordinates of atoms (shape=(na, 3), dtype=float)
        """
        return _dll.bonded_energy(*_byrefs(
            len(self.name), self.name,
            self.na, atoms.size // 3 // self.na, atoms,
            self.idx.shape[0],
            self.idx.shape[1], self.idx,
            self.par.shape[1], self.par))

    def add_gradient(self, atoms:ndarray, grad1:ndarray, delta:float=1e-4):
        """add gradient of this term to grad

        Args:
            atoms (ndarray): cartesian coordinates of atoms (shape=(na, 3), dtype=float)
            grad (ndarray): gradient by atoms (shape=(na, 3), dtype=float)
            grad2 (ndarray): gradient by lattice (shape=(3, 3), dtype=float)
        """
        _dll.bonded_gradient(*_byrefs(
            len(self.name), self.name,
            self.na, atoms.size // 3 // self.na, atoms,
            self.idx.shape[0],
            self.idx.shape[1], self.idx,
            self.par.shape[1], self.par,
            delta, grad1))

    def add_hessian(self, atoms:ndarray, hess:ndarray, delta:float=1e-4):
        """add hessian of this term to hess

        Args:
            atoms (ndarray): cartesian coordinates of atoms (shape=(na, 3), dtype=float)
            hess (ndarray): hessian by atoms (shape=(na, 3, na, 3), dtype=float)
        """
        _dll.bonded_hessian(*_byrefs(
            len(self.name), self.name,
            self.na, atoms.size // 3 // self.na, atoms,
            self.idx.shape[0],
            self.idx.shape[1], self.idx,
            self.par.shape[1], self.par,
            delta, hess))


class NonBondedTerm():
    """Force field term of non-bonded interactions

    Attributes:
        name (str): the name of force field such as "buckingham"
        na: the number of atoms
        npar: the number of force field parameters for each atom pair
        par (ndarray(shape=(na, na, npar), dtype=float)): force field parameters
        cutoff (float): cutoff length in real space
        nlmax (int): buffer size for lattice points
        nli (int): effective number of lattice points
        li (int(nlmax, 3)): the indices of the lattice points within cutoff
    """

    def __init__(self, name:str, na:int):
        self.name = name
        self.na = na
        if self.name == "lj_6_12":
            self.npar = 2
            self.nglo = 0
        elif self.name == "lj_1_6_12":
            self.npar = 3
            self.nglo = 0
        elif self.name == "buck":
            self.npar = 3
            self.nglo = 0
        elif self.name == "coul_buck":
            self.npar = 4
            self.nglo = 0
        #---- add new force field type here ----#
        else:
            raise NotImplementedError(f"Force field type not found: {name}")
        self.cutoff = 100.0
        self.nlmax = 1000
        self.nli = 0
        self.li = np.zeros((self.nlmax, 3), dtype=int)
        self.par = np.zeros((self.na, self.na, self.npar), dtype=float)
        self.glo = np.zeros(self.nglo, dtype=float)
        self.exclusion = None

    def refresh(self, atoms:ndarray, lattice:ndarray):
        nli = c_int()
        _dll.lattice_points_within_cutoff(*_byrefs(
            lattice, self.cutoff, self.nlmax, nli, self.li))
        self.nli = nli.value
        if self.nli > self.nlmax:
            self.nlmax = self.nli * 6 // 5
            self.li = np.zeros((self.nlmax, 3), dtype=int)
            self.refresh(atoms, lattice)

    def energy(self, atoms:ndarray, lattice:ndarray):
        return _dll.non_bonded_energy(*_byrefs(
            len(self.name), self.name,
            self.na, atoms.size // 3 // self.na, atoms, lattice,
            self.nli, self.li,
            self.par.shape[2], self.par,
            self.glo.shape[0], self.glo,
            self.exclusion))

    def add_gradient(self, atoms:ndarray, lattice:ndarray,
                     grad1:ndarray, grad2:ndarray, delta2:float=1e-4):
        _dll.non_bonded_gradient(*_byrefs(
            len(self.name), self.name,
            self.na, atoms.size // 3 // self.na, atoms, lattice,
            self.nli, self.li,
            self.par.shape[2], self.par,
            self.glo.shape[0], self.glo,
            self.exclusion,
            delta2, grad1, grad2))

    def add_hessian(self, atoms:ndarray, lattice:ndarray,
                    hess:ndarray, delta2=1e-4):
        _dll.non_bonded_hessian(*_byrefs(
            len(self.name), self.name,
            self.na, atoms.size // 3 // self.na, atoms, lattice,
            self.nli, self.li,
            self.par.shape[2], self.par,
            self.glo.shape[0], self.glo,
            self.exclusion,
            delta2, hess))

    def add_hessians(self, atoms:ndarray, lattice:ndarray,
                     hess11:ndarray, hess12:ndarray, hess22:ndarray, delta2:float=1e-4):
        _dll.non_bonded_hessians(*_byrefs(
            len(self.name), self.name,
            self.na, atoms.size // 3 // self.na, atoms, lattice,
            self.nli, self.li,
            self.par.shape[2], self.par,
            self.glo.shape[0], self.glo,
            self.exclusion,
            delta2, hess11, hess12, hess22))

    def add_dynamical(self, atoms:ndarray, lattice:ndarray,
                      qv:ndarray, dyna:ndarray, delta2:float=1e-4):
        _dll.non_bonded_dynamical(*_byrefs(
            len(self.name), self.name,
            self.na, atoms.size // 3 // self.na, atoms, lattice,
            self.nli, self.li,
            self.par.shape[2], self.par,
            self.glo.shape[0], self.glo,
            self.exclusion,
            delta2, qv, dyna))


class EwaldTerm():
    """Force field term of lattice sum using Ewald method: E_ij = par(i, j) * r ** (-pow)

    Attributes:
        name (str): the name of force field such as "buckingham"
        na: the number of atoms
        par (ndarray(shape=(na, na))): force field parameters
        cutoff (float): cutoff length in real space
        cutoff2 (float): cutoff length in reciprocal space
        nlmax (int): buffer size for lattice points
        nli (int): effective number of lattice points
        li (int(nlmax, 3)): the indices of the lattice points within cutoff
        nkmax (int): buffer size for lattice points
        nki (int): effective number of lattice points
        ki (int(nlmax, 3)): the indices of the lattice points within cutoff
        damp (float): damping factor
        pow (int): exponent
    """

    def __init__(self, name:str, na:int):
        self.name = name
        self.na = na
        if self.name == "ewald_6":
            self.pow = 6
        #---- add new force field type here ----#
        else:
            raise NotImplementedError(f"Force field type not found: {name}")
        self.nlmax = 1000
        self.nli = 0
        self.li = np.zeros((self.nlmax, 3), dtype=int)
        self.nkmax = 1000
        self.nki = 0
        self.ki = np.zeros((self.nkmax, 3), dtype=int)
        self.par = np.zeros((self.na, self.na))
        self.exclusion = None

    def refresh(self, atoms:ndarray, lattice:ndarray):
        scaler = 10.0
        self.damp = float(abs(np.linalg.det(lattice))**(1.0/3.0) / math.sqrt(math.pi))
        self.cutoff = scaler * self.damp
        self.cutoff2 = scaler * 2 / self.damp

        nli = c_int()
        _dll.lattice_points_within_cutoff(*_byrefs(
            lattice, self.cutoff, self.nlmax, nli, self.li))
        self.nli = nli.value
        if self.nli > self.nlmax:
            self.nlmax = self.nli * 6 // 5
            self.li = np.zeros((self.nlmax, 3), dtype=int)
            self.refresh(atoms, lattice)

        nki = c_int()
        pkv = csp.reciprocal_lattice(lattice)
        _dll.lattice_points_within_cutoff(*_byrefs(
            pkv, self.cutoff2, self.nkmax, nki, self.ki))
        self.nki = nki.value
        
        if self.nli > self.nlmax or self.nki > self.nkmax:
            if self.nli > self.nlmax:
                self.nlmax = self.nli * 6 // 5
                self.li = np.zeros((self.nlmax, 3), dtype=int)
            if self.nki > self.nkmax:
                self.nkmax = self.nki * 6 // 5
                self.ki = np.zeros((self.nkmax, 3), dtype=int)
            self.refresh(atoms, lattice)

    def energy(self, atoms:ndarray, lattice:ndarray):
        return _dll.ewald_energy(*_byrefs(
            self.pow, self.na, atoms.size // 3 // self.na, atoms, lattice,
            self.nli, self.li, self.nki, self.ki, self.par, self.damp, self.exclusion))

    def add_gradient(self, atoms:ndarray, lattice:ndarray,
                     grad1:ndarray, grad2:ndarray, delta2:float=1e-4):
        pass
        # _dll.non_bonded_gradient(*_byrefs(
        #     len(self.name), self.name,
        #     self.na, atoms.size // 3 // self.na, atoms, lattice,
        #     self.nli, self.li,
        #     self.par.shape[2], self.par,
        #     self.glo.shape[0], self.glo,
        #     self.exclusion,
        #     delta2, grad1, grad2))

    def add_hessian(self, atoms:ndarray, lattice:ndarray,
                    hess:ndarray, delta2=1e-4):
        pass
        # _dll.non_bonded_hessian(*_byrefs(
        #     len(self.name), self.name,
        #     self.na, atoms.size // 3 // self.na, atoms, lattice,
        #     self.nli, self.li,
        #     self.par.shape[2], self.par,
        #     self.glo.shape[0], self.glo,
        #     self.exclusion,
        #     delta2, hess))

    def add_hessians(self, atoms:ndarray, lattice:ndarray,
                     hess11:ndarray, hess12:ndarray, hess22:ndarray, delta2:float=1e-4):
        pass
        # _dll.non_bonded_hessians(*_byrefs(
        #     len(self.name), self.name,
        #     self.na, atoms.size // 3 // self.na, atoms, lattice,
        #     self.nli, self.li,
        #     self.par.shape[2], self.par,
        #     self.glo.shape[0], self.glo,
        #     self.exclusion,
        #     delta2, hess11, hess12, hess22))

    def add_dynamical(self, atoms:ndarray, lattice:ndarray,
                      qv:ndarray, dyna:ndarray, delta2:float=1e-4):
        pass
        # _dll.non_bonded_dynamical(*_byrefs(
        #     len(self.name), self.name,
        #     self.na, atoms.size // 3 // self.na, atoms, lattice,
        #     self.nli, self.li,
        #     self.par.shape[2], self.par,
        #     self.glo.shape[0], self.glo,
        #     self.exclusion,
        #     delta2, qv, dyna))


def set_openmp(enabled:bool):
    global _dll
    if enabled:
        _dll = _load_dll('forcefield_omp.dll', os.path.join(os.path.dirname(sys.modules[__name__].__file__), "fortran"))
    else:
        _dll = _load_dll('forcefield.dll', os.path.join(os.path.dirname(sys.modules[__name__].__file__), "fortran"))


# def _empty_gradients(n):
#     """Create and returns empty gradient arrays, *(grad, grad1, grad2)*.

#     Note:
#         *grad1* and *grad2* is the views of submatrices of *grad*, and share memory."""
#     grad = np.zeros((n+3, 3))
#     grad1 = grad[:-3, :]
#     grad2 = grad[-3:, :]
#     return grad, grad1, grad2


# def _empty_hessians(n):
#     """
#     Create and returns empty hessian arrays, *(hess, hess11, hess12, hess22)*.
        
#     Note:
#         *hess11*, *hess12* and *hess22* are the submatrices of *hess*.
#         However, these matrices do not share memory.
#         Use *_join_hessians* function to copy *hess11*, *hess12* and *hess22* to *hess*.
#     """
#     hess = np.zeros((n+3, 3, n+3, 3))
#     hess11 = np.zeros((n, 3, n, 3))
#     hess12 = np.zeros((n, 3, 3, 3))
#     hess22 = np.zeros((3, 3, 3, 3))
#     return hess, hess11, hess12, hess22


# def _join_hessians(hess11, hess12, hess22, hess=None):
#     """Copy the submatrices *hess11*, *hess12* and *hess22* to *hess*."""
#     if hess is None:
#         n = hess11.shape[0]
#         hess = np.zeros((n+3, 3, n+3, 3))
#     hess[:-3, :, :-3, :] = hess11
#     hess[:-3, :, -3:, :] = hess12
#     hess[-3:, :, :-3, :] = hess12.transpose((2, 3, 0, 1))
#     hess[-3:, :, -3:, :] = hess22
#     return hess


def _byrefs(*args):
    refs = []
    for a in args:
        t = type(a)
        if t is int:
            refs.append(byref(c_int(a)))
        elif t is float:
            refs.append(byref(c_double(a)))
        elif t is ndarray:
            refs.append(a.T)
        elif t is str:
            refs.append(create_string_buffer(a.encode("utf-8")))
        else:
            refs.append(a)
    return refs


def _load_dll(filename, directory):

    dll = np.ctypeslib.load_library(filename, directory)

    i = POINTER(c_int)
    f = POINTER(c_double)
    ia = ndpointer(dtype=np.int)
    fa = ndpointer(dtype=np.float)
    c = c_char_p

    dll.bonded_energy.argtypes       = i, c, i, i, fa, i, i, ia, i, fa
    dll.bonded_energy.restype        = c_double
    dll.bonded_gradient.argtypes     = i, c, i, i, fa, i, i, ia, i, fa, f, fa
    dll.bonded_gradient.restype      = c_void_p
    dll.bonded_hessian.argtypes      = i, c, i, i, fa, i, i, ia, i, fa, f, fa
    dll.bonded_hessian.restype       = c_void_p
    dll.non_bonded_energy.argtypes   = i, c, i, i, fa, fa, i, ia, i, fa, i, fa, fa
    dll.non_bonded_energy.restype    = c_double
    dll.non_bonded_gradient.argtypes = i, c, i, i, fa, fa, i, ia, i, fa, i, fa, fa, f, fa, fa
    dll.non_bonded_gradient.restype  = c_void_p
    dll.non_bonded_hessian.argtypes  = i, c, i, i, fa, fa, i, ia, i, fa, i, fa, fa, f, fa
    dll.non_bonded_hessian.restype   = c_void_p
    dll.non_bonded_hessians.argtypes = i, c, i, i, fa, fa, i, ia, i, fa, i, fa, fa, f, fa, fa, fa
    dll.non_bonded_hessians.restype  = c_void_p
    dll.ewald_energy.argtypes   = i, i, i, fa, fa, i, ia, i, ia, fa, f, fa
    dll.ewald_energy.restype    = c_double
    dll.lattice_points_within_cutoff.argtypes = fa, f, i, i, ia
    dll.lattice_points_within_cutoff.restype = c_void_p

    return dll


# load forcefield.dll when this module is imported
set_openmp(False)
