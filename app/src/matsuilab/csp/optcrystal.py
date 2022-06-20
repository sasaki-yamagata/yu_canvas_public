"""OptCrystal class for crystal structure optimization.

Examples:
    This example code reads the initial structure "initial.cif" and a force field file "sample.ff", optimizes the structure, and save the optimized structure to "final.cif". Trajectory during optimization is saved as "trajectory.cif".

    >>> ff = ForceField.open("sample.ff")
    >>> crystal = OptCrystal.read_cif("initial.cif")
    >>> crystal.minimize_lattice_energy(ff, trajectory="trajectory.cif")
    >>> crystal.write_cif("final.cif")
"""

import math
from logging import Logger, getLogger
from typing import Sequence

import numpy as np
from matsuilab import atpro, csp
from matsuilab.csp.forcefield import ForceField
from numpy import ndarray


class OptCrystal():
    """Crystal for structural optimization.

    Notes:
        This class does not support topological changes such as adding atoms or bonds. The number of atoms and bonds must be given at the constructor and must not be changed after that.

    Attributes:
        name: name of this crystal (str)
        natoms: the number of atoms (int)
        nbonds: the number of bonds (int)
        numbers: list of atomic numbers (ndarray(shape=(natoms), dtype=int))
        coordinates: x, y, z coordinates of all atoms and lattice vectors (ndarray(shape=(natoms+3, 3), dtype=float))
        atoms: view of atom part of coordinates (ndarray(shape=(natoms, 3), dtype=float))
        lattice: view of lattice part of coordinates (ndarray(shape=(3, 3), dtype=float))
        bonds: bonded atom indices and bond type (ndarray(shape=(nbonds, 3), dtype=int))
        spacegroup: space group id (int)
    """

    def __init__(self, name:str="noname", natoms:int=0, nbonds:int=0):
        self.name : str = name
        self.natoms : int = natoms
        self.nbonds : int = nbonds
        self.numbers : ndarray = np.zeros(natoms, dtype=int)
        self.coordinates : ndarray = np.zeros((natoms+3, 3), dtype=float)
        self.atoms : ndarray = self.coordinates[:-3, :]
        self.lattice : ndarray = self.coordinates[-3:, :]
        self.bonds : ndarray = np.zeros((nbonds, 3), dtype=int)
        self.spacegroup : int = 1
        self.adp : ndarray = None

    @classmethod
    def read_cif(cls, file:str=None, content:str=None):
        """Read cif file or its content and returns Crystal object. This function requires OpenBabel."""
        from openbabel import openbabel as ob

        mol = ob.OBMol()
        conv = ob.OBConversion()
        conv.SetInFormat("cif")
        if file is not None:
            conv.ReadFile(mol, file)
        elif content is not None:
            conv.ReadString(mol, content)
        else:
            raise Exception("Either file or content must be given.")

        self = cls(name=mol.GetTitle(), natoms=mol.NumAtoms(),
                   nbonds=mol.NumBonds())

        # get atom coordinates
        for i, a in enumerate(ob.OBMolAtomIter(mol)):
            self.numbers[i] = a.GetAtomicNum()
            self.atoms[i, :] = a.GetX(), a.GetY(), a.GetZ()

        # get bond information
        for i, b in enumerate(ob.OBMolBondIter(mol)):
            if b.IsAromatic():
                self.bonds[i, :] = b.GetBeginAtomIdx(), b.GetEndAtomIdx(), 4
            else:
                self.bonds[i, :] = b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondOrder()

        # get lattice and space group
        cell = ob.toUnitCell(mol.GetData(ob.UnitCell))
        for i, vec in enumerate(cell.GetCellVectors()):
            self.lattice[i, :] = vec.GetX(), vec.GetY(), vec.GetZ()
        self.spacegroup = cell.GetSpaceGroup().GetId()

        return self

    def write_cif(self, file:str=None, name:str=None, labels:Sequence[str]=None, append:bool=False) -> str:
        """write cif file or returns its content

        Args:
            name: name of data block
            labels: atom labels, optional
            file: cif file name
            append: whether the data is added to existing file
        """
        if name is None:
            name = self.name
        return csp.write_cif(name=name, atoms=self.atoms, lattice=self.lattice,
            numbers=self.numbers, labels=labels, spacegroup=1, bonds=self.bonds,
            adp=self.adp, file=file, append=append)

    def minimize_lattice_energy(self, ff:ForceField, opt_lattice:bool=True,
        delta:float=1e-4, newton:bool=True, conjugate:bool=False,
        maxcyc:int=1000, xtol:float=1e-3, ytol:float=1e-9, gtol:float=1e-3,
        alpha0:float=1.0, maxstep:float=1.0, linecyc:int=10,
        irefresh:int=10, ireport:int=1, trajectory:str="trajectory.cif",
        logger:Logger=getLogger()) -> csp.DictLike:
        """perform lattice energy minimization. atoms[0, :], lattice[0, 1], lattice[0, 2], lattice[1, 2] are unchanged.
        
        Args:
            ff: force field
            opt_lattice: whether lattice vectors are optimized
            newton: whether modified newton method is used
            conjugate: whether conjugate gradient method is used
            maxcyc: maximum number of cycles
            xtol: tolerance for the norm of delta-x
            ytol: tolerance for delta-y
            gtol: tolerance for the norm of gradient
            alpha0: initial guess of step in line search
            maxstep: upper limit of one step
            linecyc: maximum number of cycles for line search. If linecyc is 0 or None, the given alpha is used as is.
            trajectory: cif file name for trajectory
            irefresh: interval steps for refreshing
            ireport: interval steps for printing report
            """

        logger = logger.getChild("lem")

        self.write_cif(name=f"initial", file=trajectory)

        if opt_lattice:
            mask = np.array(tuple([i < 3 for i in range(self.atoms.size)]) + (False, True, True, False, False, True, False, False, False), dtype=bool)
            x0 = self.coordinates
        else:
            mask = np.array(tuple([i < 3 for i in range(self.atoms.size)]), dtype=bool)
            x0 = self.atoms

        if newton:
            hfun = self._hfun(ff, opt_lattice, delta)
        else:
            hfun = None

        # define criteria
        criteria = []
        criteria.append(csp.y_criteria(ytol))
        criteria.append(csp.x_criteria(xtol))
        criteria.append(csp.g_criteria(gtol))
        criteria.append(csp.h_criteria())

        # define callback
        callback = []
        callback.append(self._rfun(ff, opt_lattice, irefresh))
        callback.append(self._mod_callback(index=np.reshape(np.arange(self.natoms), (self.natoms // ff.na, ff.na)), interval=irefresh))
        callback.append(csp.default_report(ireport))
        callback.append(self._trajectory_callback(file=trajectory, interval=ireport))
            
        result = csp.minimize(fun=self._fun(ff, opt_lattice), x0=x0, mask=mask,
            gfun=self._gfun(ff, opt_lattice, delta), hfun=hfun, conjugate=conjugate, 
            maxcyc=maxcyc, alpha0=alpha0, maxstep=maxstep, linecyc=linecyc,
            criteria=criteria, callback=callback, logger=logger)

        self.write_cif(name=f"final", file=trajectory, append=True)

        return result

    def minimize_free_energy(self, ff:ForceField, temp:float,
        delta:float=1e-4, newton:bool=True, conjugate:bool=True,
        maxcyc:int=1000, xtol:float=1e-3, ytol:float=1e-9, gtol:float=1e-3,
        alpha0:float=1e-3, maxstep:float=0.1, linecyc:int=10,
        irefresh:int=1, ireport:int=1, trajectory:str="trajectory.cif",
        logger:Logger=getLogger()) -> csp.DictLike:
        """perform free energy minimization.

        Args:
            ff: force field
            opt_lattice: whether lattice vectors are optimized
            newton: whether modified newton method is used
            conjugate: whether conjugate gradient method is used
            maxcyc: maximum number of cycles
            xtol: tolerance for the norm of delta-x
            ytol: tolerance for delta-y
            gtol: tolerance for the norm of gradient
            alpha0: initial guess of step in line search
            maxstep: upper limit of one step
            linecyc: maximum number of cycles for line search. If linecyc is 0 or None, the given alpha is used as is.
            trajectory: cif file name for trajectory
            irefresh: interval steps for refreshing
            ireport: interval steps for printing report
        """

        logger = logger.getChild("fem")

        self.write_cif(name=f"initial", file=trajectory)

        n = self.natoms
        mask = np.array((False, True, True, False, False, True, False, False, False), dtype=bool)

        buffer = np.array(self.atoms)
        hess11 = np.zeros((n, 3, n, 3))

        def fun(x):
            buffer[:, :] = self.atoms
            self.lattice[:, :] = np.reshape(x, (3, 3))

            self.minimize_lattice_energy(ff=ff, opt_lattice=False,
                delta=delta, newton=newton, conjugate=conjugate,
                maxcyc=maxcyc, xtol=xtol, ytol=ytol, gtol=gtol,
                alpha0=alpha0, maxstep=maxstep, linecyc=linecyc,
                irefresh=0, ireport=1, trajectory="trajectory_sub.cif", logger=logger)

            lattice_energy = ff.energy(self.atoms, self.lattice)

            ff.calc_hessian(self.atoms, self.lattice, hess11, delta)
            frequencies, _ = csp.calc_phonons(mass=atpro.WEIGHTS[self.numbers], hess11=hess11, temp=temp, eunit="kcal/mol", logger=logger)
            phonon_energy = csp.phonon_free_energy(frequencies, temp)

            self.atoms[:, :] = buffer

            return lattice_energy + phonon_energy

        def update_atoms(o:csp.DictLike):
            self.minimize_lattice_energy(ff=ff, opt_lattice=False,
                delta=delta, newton=newton, conjugate=conjugate,
                maxcyc=maxcyc, xtol=xtol, ytol=ytol, gtol=gtol,
                alpha0=alpha0, maxstep=maxstep, linecyc=linecyc,
                irefresh=0, ireport=0, trajectory=None, logger=logger)
            return True

        # define criteria
        criteria = []
        criteria.append(csp.y_criteria(ytol))
        criteria.append(csp.x_criteria(xtol))
        criteria.append(csp.g_criteria(gtol))

        # define callback
        callback = []
        callback.append(self._rfun2(ff, irefresh))
        callback.append(update_atoms)
        callback.append(self._mod_callback(index=np.reshape(np.arange(self.natoms), (self.natoms // ff.na, ff.na)), interval=irefresh))
        callback.append(csp.default_report(ireport))
        callback.append(self._trajectory_callback(file=trajectory, interval=ireport))

        result = csp.minimize(fun=fun, x0=self.lattice, mask=mask, gfun=-delta,
            conjugate=True, maxcyc=maxcyc, 
            alpha0=alpha0, maxstep=maxstep, linecyc=linecyc,
            criteria=criteria, callback=callback, logger=logger)

        self.write_cif(name=f"final", file=trajectory, append=True)

        return result

    def calc_phonons(self, ff:ForceField, temp:float=0.0, eunit="kcal/mol"):
        """Returns (frequences, modes) and set anisotropic displacement parameters at self.adp"""
        hess11 = np.zeros((self.natoms, 3, self.natoms, 3))
        ff.calc_hessian(self.atoms, self.lattice, hess11)
        frequencies, modes = csp.calc_phonons(atpro.WEIGHTS[self.numbers], hess11, temp=temp, eunit=eunit)
        self.adp = csp.anisotropic_displacement_parameters(modes)
        return frequencies, modes

    def write_vibration_cif(self, file:str, frequencies:ndarray, modes:ndarray, scaler:float=1.0, div:int=20):
        original = np.array(self.atoms)
        for i, (f, mode) in enumerate(zip(frequencies, modes)):
            for j in range(div):
                self.atoms[:] = original + scaler * math.sin(2.0 * math.pi * j / div) * mode
                self.write_cif(file, name=f"mode{i}_{f*1e-9:.0f}GHz_{j}", append=(i>0 or j>0))

    def _fun1(self, ff:ForceField):
        return lambda x: ff.energy(x, self.lattice)
    
    def _fun2(self, ff:ForceField):
        return lambda x: ff.energy(x[:-9], x[-9:])

    def _fun(self, ff:ForceField, opt_lattice:bool):
        if opt_lattice:
            return self._fun2(ff)
        else:
            return self._fun1(ff)

    def _gfun1(self, ff:ForceField, delta=1e-4):
        gradients = np.zeros((self.natoms+3, 3))
        grad1 = gradients[:-3, :] # view
        grad2 = gradients[-3:, :] # view

        def gfun(x):
            ff.calc_gradients(x, self.lattice, grad1, grad2, delta)
            return np.reshape(grad1, grad1.size)

        return gfun

    def _gfun2(self, ff:ForceField, delta=1e-4):
        gradients = np.zeros((self.natoms+3, 3))
        grad1 = gradients[:-3, :] # view
        grad2 = gradients[-3:, :] # view

        def gfun(x):
            ff.calc_gradients(x[:-9], x[-9:], grad1, grad2, delta)
            return np.reshape(gradients, gradients.size)

        return gfun

    def _gfun(self, ff:ForceField, opt_lattice:bool, delta=1e-4):
        if opt_lattice:
            return self._gfun2(ff, delta)
        else:
            return self._gfun1(ff, delta)

    def _hfun1(self, ff:ForceField, delta=1e-4):
        hess11 = np.zeros((self.natoms, 3, self.natoms, 3))
        def hfun(x):
            ff.calc_hessian(x, self.lattice, hess11, delta)
            return np.reshape(hess11, (self.natoms*3, self.natoms*3))

        return hfun

    def _hfun2(self, ff:ForceField, delta=1e-4):
        hessian = np.zeros((self.natoms+3, 3, self.natoms+3, 3))
        hess11 = np.zeros((self.natoms, 3, self.natoms, 3))
        hess12 = np.zeros((self.natoms, 3, 3, 3))
        hess22 = np.zeros((3, 3, 3, 3))

        def hfun(x):
            ff.calc_hessians(x[:-9], x[-9:], hess11, hess12, hess22, delta)
            hessian[:-3, :, :-3, :] = hess11
            hessian[:-3, :, -3:, :] = hess12
            hessian[-3:, :, :-3, :] = hess12.transpose((2, 3, 0, 1))
            hessian[-3:, :, -3:, :] = hess22
            return np.reshape(hessian, (x.size, x.size))

        return hfun

    def _hfun(self, ff:ForceField, opt_lattice:bool, delta=1e-4):
        if opt_lattice:
            return self._hfun2(ff, delta)
        else:
            return self._hfun1(ff, delta)

    def _rfun1(self, ff:ForceField, interval:int=10):
        def rfun(o:csp.DictLike):
            if interval > 0 and o.cycle % interval == 0:
                ff.refresh(o.x, self.lattice)
                return True
            else:
                return False
        return rfun

    def _rfun2(self, ff:ForceField, interval:int=10):
        def rfun(o:csp.DictLike):
            if interval > 0 and o.cycle % interval == 0:
                ff.refresh(o.x[:-9], o.x[-9:])
                return True
            else:
                return False
        return rfun

    def _rfun3(self, ff:ForceField, interval:int=10):
        def rfun(o:csp.DictLike):
            if interval > 0 and o.cycle % interval == 0:
                ff.refresh(self.atoms, o.x)
                return True
            else:
                return False
        return rfun

    def _rfun(self, ff:ForceField, opt_lattice:bool, interval:int=10):
        if opt_lattice:
            return self._rfun2(ff, interval)
        else:
            return self._rfun1(ff, interval)

    def _trajectory_callback(self, file:str, interval:int=10):
        def tfun(o:csp.DictLike):
            if interval > 0 and o.cycle % interval == 0:
                self.write_cif(name=str(o.cycle), file=file, append=True)
            return False
        return tfun

    def mod_molecules(self, index:Sequence[Sequence[int]]):
        for idx in index:
            center = np.average(self.atoms[idx, :], axis=0)
            cell = np.floor(csp.to_fractional(center, self.lattice))
            self.atoms[idx, :] -= csp.to_cartesian(cell, self.lattice)

    def _mod_callback(self, index:Sequence[Sequence[int]], interval:int=10):
        def mfun(o:csp.DictLike):
            if interval > 0 and o.cycle % interval == 0:
                self.mod_molecules(index)
            return True
        return mfun

    # def minimize_free_energy(self, ff:ForceField,
    #     conjugate:bool=False,
    #     maxcyc:int=1000, xtol:float=None, ytol:float=1e-9, gtol:float=None,
    #     alpha0:float=1.0, maxstep:float=1.0, subcyc:int=1000, linecyc:int=10, trajectory:str=None,
    #     irefresh:int=10, ireport:int=1):
    #     pass
