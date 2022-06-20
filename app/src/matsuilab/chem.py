"""
Handling molecule data

Main class is FlexMol, which provides a flexible way of handling molecules.
FlexMol objects can be converted from and to various data formats.
FlexMol also supports some operations such as add_atom, add_mol, split, attach_atom, gen3d, translate, and rotate.

Example:
    FlexMol object can be created from another object or a file by the constructor

    >>> mol1 = FlexMol("sample.mol")         # from a file
    >>> mol2 = FlexMol(csd_molecule)         # from another object
    >>> mol3 = FlexMol("c1ccccc1") # from SMILES

    FlexMol object can be converted to another object or file

    >>> mol1.to("sample.xyz")      # to a file
    >>> csd    = mol2.to("csd")    # to another object
    >>> smiles = mol3.to("smiles") # to SMILES

    Direct conversion between file formats is possible as follows.

    >>> FlexMol("input.mol").to("output.mol2") # file to file
    >>> rdmol = FlexMol(csdmol).to("rd")       # CSD to RDKit

Supported classes:
    - OpenBabel OBMol (format name: "ob")
    - RDKit Mol (format name: "rd")
    - CSD Molecule (format name: "csd")

Supported file formats:
    - see SUPPORTED_INPUT_FORMATS and SUPPORTED_OUTPUT_FORMATS
"""

from __future__ import annotations

import datetime
import math
import os
import random
import subprocess
import sys
from collections import abc
from typing import Any, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from matsuilab import atpro

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdmolfiles
except:
    pass

try:
    from ccdc import io
except:
    pass

try:
    from openbabel import openbabel as ob
except:
    pass


def _angle(vec1, vec2):
    """
    returns the angle between two vectors (unit: radian)
    """
    return math.acos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


def _angle_degree(vec1, vec2):
    """
    returns the angle between two vectors (unit: degree)
    """
    return _angle(vec1, vec2) * 180.0 / math.pi


_HM_NAMES = ('', 'P 1', 'P -1', 'P 1 2 1', 'P 1 21 1', 'C 1 2 1', 'P 1 m 1', 'P 1 c 1', 'C 1 m 1', 'C 1 c 1', 'P 1 2/m 1',
    'P 1 21/m 1', 'C 1 2/m 1', 'P 1 2/c 1', 'P 1 21/c 1', 'C 1 2/c 1', 'P 2 2 2', 'P 2 2 21', 'P 21 21 2', 'P 21 21 21', 'C 2 2 21',
    'C 2 2 2', 'F 2 2 2', 'I 2 2 2', 'I 21 21 21', 'P m m 2', 'P m c 21', 'P c c 2', 'P m a 2', 'P c a 21', 'P n c 2',
    'P m n 21', 'P b a 2', 'P n a 21', 'P n n 2', 'C m m 2', 'C m c 21', 'C c c 2', 'A m m 2', 'A b m 2', 'A m a 2',
    'A b a 2', 'F m m 2', 'F d d 2', 'I m m 2', 'I b a 2', 'I m a 2', 'P m m m', 'P n n n:1', 'P c c m', 'P b a n:1',
    'P m m a', 'P n n a', 'P m n a', 'P c c a', 'P b a m', 'P c c n', 'P b c m', 'P n n m', 'P m m n:1', 'P b c n',
    'P b c a', 'P n m a', 'C m c m', 'C m c a', 'C m m m', 'C c c m', 'C m m a', 'C c c a:2', 'F m m m', 'F d d d:1',
    'I m m m', 'I b a m', 'I b c a', 'I m m a', 'P 4', 'P 41', 'P 42', 'P 43', 'I 4', 'I 41',
    'P -4', 'I -4', 'P 4/m', 'P 42/m', 'P 4/n:1', 'P 42/n:1', 'I 4/m', 'I 41/a:1', 'P 4 2 2', 'P 42 1 2',
    'P 41 2 2', 'P 41 21 2', 'P 42 2 2', 'P 42 21 2', 'P 43 2 2', 'P 43 21 2', 'I 4 2 2', 'I 41 2 2', 'P 4 m m', 'P 4 b m',
    'P 42 c m', 'P 42 n m', 'P 4 c c', 'P 4 n c', 'P 42 m c', 'P 42 b c', 'I 4 m m', 'I 4 c m', 'I 41 m d', 'I 41 c d',
    'P -4 2 m', 'P -4 2 c', 'P -4 21 m', 'P -4 21 c', 'P -4 m 2', 'P -4 c 2', 'P -4 b 2', 'P -4 n 2', 'I -4 m 2', 'I -4 c 2',
    'I -4 2 m', 'I -4 2 d', 'P 4/m m m', 'P 4/m c c', 'P 4/n b m:1', 'P 4/n n c:1', 'P 4/m b m', 'P 4/m n c', 'P 4/n m m:1', 'P 4/n c c:1',
    'P 42/m m c', 'P 42/m c m', 'P 42/n b c:1', 'P 42/n n m:1', 'P 42/m b c', 'P 42/m n m', 'P 42/n m c:1', 'P 42/n c m:1', 'I 4/m m m', 'I 4/m c m',
    'I 41/a m d:1', 'I 41/a c d:1', 'P 3', 'P 31', 'P 32', 'R 3:H', 'P -3', 'R -3:H', 'P 3 1 2', 'P 3 2 1',
    'P 31 1 2', 'P 31 2 1', 'P 32 1 2', 'P 32 2 1', 'R 32:H', 'P 3 m 1', 'P 3 1 m', 'P 3 c 1', 'P 3 1 c', 'R 3 m:H',
    'R 3 c:H', 'P -3 1 m', 'P -3 1 c', 'P -3 m 1', 'P -3 c 1', 'R -3 m:H', 'R -3 c:H', 'P 6', 'P 61', 'P 65',
    'P 62', 'P 64', 'P 63', 'P -6', 'P 6/m', 'P 63/m', 'P 6 2 2', 'P 61 2 2', 'P 65 2 2', 'P 62 2 2',
    'P 64 2 2', 'P 63 2 2', 'P 6 m m', 'P 6 c c', 'P 63 c m', 'P 63 m c', 'P -6 m 2', 'P -6 c 2', 'P -6 2 m', 'P -6 2 c',
    'P 6/m m m', 'P 6/m c c', 'P 63/m c m', 'P 63/m m c', 'P 2 3', 'F 2 3', 'I 2 3', 'P 21 3', 'I 21 3', 'P m 3',
    'P n 3:1', 'F m 3', 'F d 3:1', 'I m 3', 'P a 3', 'I a 3', 'P 4 3 2', 'P 42 3 2', 'F 4 3 2', 'F 41 3 2',
    'I 4 3 2', 'P 43 3 2', 'P 41 3 2', 'I 41 3 2', 'P -4 3 m', 'F -4 3 m', 'I -4 3 m', 'P -4 3 n', 'F -4 3 c', 'I -4 3 d',
    'P m 3 m', 'P n 3 n:1', 'P m 3 n', 'P n 3 m:1', 'F m 3 m', 'F m 3 c', 'F d 3 m:1', 'F d 3 c:1', 'I m 3 m', 'I a 3 d')

_HALL_NAMES = ('', 'P 1', '-P 1', 'P 2y', 'P 2yb', 'C 2y', 'P -2y', 'P -2yc', 'C -2y', 'C -2yc', '-P 2y',
    '-P 2yb', '-C 2y', '-P 2yc', '-P 2ybc', '-C 2yc', 'P 2 2', 'P 2c 2', 'P 2 2ab', 'P 2ac 2ab', 'C 2c 2',
    'C 2 2', 'F 2 2', 'I 2 2', 'I 2b 2c', 'P 2 -2', 'P 2c -2', 'P 2 -2c', 'P 2 -2a', 'P 2c -2ac', 'P 2 -2bc',
    'P 2ac -2', 'P 2 -2ab', 'P 2c -2n', 'P 2 -2n', 'C 2 -2', 'C 2c -2', 'C 2 -2c', 'A 2 -2', 'A 2 -2c', 'A 2 -2a',
    'A 2 -2ac', 'F 2 -2', 'F 2 -2d', 'I 2 -2', 'I 2 -2c', 'I 2 -2a', '-P 2 2', 'P 2 2 -1n', '-P 2 2c', 'P 2 2 -1ab',
    '-P 2a 2a', '-P 2a 2bc', '-P 2ac 2', '-P 2a 2ac', '-P 2 2ab', '-P 2ab 2ac', '-P 2c 2b', '-P 2 2n', 'P 2 2ab -1ab', '-P 2n 2ab',
    '-P 2ac 2ab', '-P 2ac 2n', '-C 2c 2', '-C 2bc 2', '-C 2 2', '-C 2 2c', '-C 2b 2', '-C 2b 2bc', '-F 2 2', 'F 2 2 -1d',
    '-I 2 2', '-I 2 2c', '-I 2b 2c', '-I 2b 2', 'P 4', 'P 4w', 'P 4c', 'P 4cw', 'I 4', 'I 4bw',
    'P -4', 'I -4', '-P 4', '-P 4c', 'P 4ab -1ab', 'P 4n -1n', '-I 4', 'I 4bw -1bw', 'P 4 2', 'P 4ab 2ab',
    'P 4w 2c', 'P 4abw 2nw', 'P 4c 2', 'P 4n 2n', 'P 4cw 2c', 'P 4nw 2abw', 'I 4 2', 'I 4bw 2bw', 'P 4 -2', 'P 4 -2ab',
    'P 4c -2c', 'P 4n -2n', 'P 4 -2c', 'P 4 -2n', 'P 4c -2', 'P 4c -2ab', 'I 4 -2', 'I 4 -2c', 'I 4bw -2', 'I 4bw -2c',
    'P -4 2', 'P -4 2c', 'P -4 2ab', 'P -4 2n', 'P -4 -2', 'P -4 -2c', 'P -4 -2ab', 'P -4 -2n', 'I -4 -2', 'I -4 -2c',
    'I -4 2', 'I -4 2bw', '-P 4 2', '-P 4 2c', 'P 4 2 -1ab', 'P 4 2 -1n', '-P 4 2ab', '-P 4 2n', 'P 4ab 2ab -1ab', 'P 4ab 2n -1ab',
    '-P 4c 2', '-P 4c 2c', 'P 4n 2c -1n', 'P 4n 2 -1n', '-P 4c 2ab', '-P 4n 2n', 'P 4n 2n -1n', 'P 4n 2ab -1n', '-I 4 2', '-I 4 2c',
    'I 4bw 2bw -1bw', 'I 4bw 2aw -1bw', 'P 3', 'P 31', 'P 32', 'R 3', '-P 3', '-R 3', 'P 3 2', 'P 3 2"',
    'P 31 2c (0 0 1)', 'P 31 2"', 'P 32 2c (0 0 -1)', 'P 32 2"', 'R 3 2"', 'P 3 -2"', 'P 3 -2', 'P 3 -2"c', 'P 3 -2c', 'R 3 -2"',
    'R 3 -2"c', '-P 3 2', '-P 3 2c', '-P 3 2"', '-P 3 2"c', '-R 3 2"', '-R 3 2"c', 'P 6', 'P 61', 'P 65',
    'P 62', 'P 64', 'P 6c', 'P -6', '-P 6', '-P 6c', 'P 6 2', 'P 61 2 (0 0 -1)', 'P 65 2 (0 0 1)', 'P 62 2c (0 0 1)',
    'P 64 2c (0 0 -1)', 'P 6c 2c', 'P 6 -2', 'P 6 -2c', 'P 6c -2', 'P 6c -2c', 'P -6 2', 'P -6c 2', 'P -6 -2', 'P -6c -2c',
    '-P 6 2', '-P 6 2c', '-P 6c 2', '-P 6c 2c', 'P 2 2 3', 'F 2 2 3', 'I 2 2 3', 'P 2ac 2ab 3', 'I 2b 2c 3', '-P 2 2 3',
    'P 2 2 3 -1n', '-F 2 2 3', 'F 2 2 3 -1d', '-I 2 2 3', '-P 2ac 2ab 3', '-I 2b 2c 3', 'P 4 2 3', 'P 4n 2 3', 'F 4 2 3', 'F 4d 2 3',
    'I 4 2 3', 'P 4acd 2ab 3', 'P 4bd 2ab 3', 'I 4bd 2c 3', 'P -4 2 3', 'F -4 2 3', 'I -4 2 3', 'P -4n 2 3', 'F -4c 2 3', 'I -4bd 2c 3',
    '-P 4 2 3', 'P 4 2 3 -1n', '-P 4n 2 3', 'P 4n 2 3 -1n', '-F 4 2 3', '-F 4c 2 3', 'F 4d 2 3 -1d', 'F 4d 2 3 -1cd', '-I 4 2 3', '-I 4bd 2c 3')

_BOND_TYPES = ("", "sing", "doub", "trip", "arom")

class Atom():
    """
    atom in a FlexMol object. This object should be created by FlexMol.add_atom().

    Attributes:
        number: atomic number
        xyz: cartesian coordinate
        charge: formal charge
    """

    _table = pd.read_csv(os.path.join(os.path.dirname(
        sys.modules[__name__].__file__), "element_properties.csv"))

    def __init__(self, number: int, x: float = 0.0, y: float = 0.0, z: float = 0.0, charge: int = 0):
        self.number: int
        if type(number) is str:
            self.number = atpro.SYMBOLS.index(number)
        else:
            self.number = int(number)
        self.xyz: np.ndarray[float] = np.array(
            (float(x), float(y), float(z)))
        self.charge: int = int(charge)

    @property
    def symbol(self) -> str:
        return atpro.SYMBOLS[self.number]

    @symbol.setter
    def symbol(self, symbol: str):
        self.number = str(symbol)

    @property
    def x(self) -> float:
        """equivalent to xyz[0]"""
        return self.xyz[0]

    @x.setter
    def x(self, x: float):
        self.xyz[0] = float(x)

    @property
    def y(self) -> float:
        """equivalent to xyz[1]"""
        return self.xyz[1]

    @y.setter
    def y(self, y: float):
        self.xyz[1] = float(y)

    @property
    def z(self) -> float:
        """equivalent to xyz[2]"""
        return self.xyz[2]

    @z.setter
    def z(self, z: float):
        self.xyz[2] = float(z)

    @property
    def symbol(self) -> str:
        """element symbol"""
        return self._table.at[self.number, "symbol"]

    @property
    def period(self) -> int:
        """period in periodic table"""
        return self._table.at[self.number, "period"]

    @property
    def group(self) -> int:
        """group in periodic table"""
        return self._table.at[self.number, "group"]

    @property
    def valence_electrons(self) -> int:
        """the number of valence electrons"""
        return self._table.at[self.number, "valence_electrons"]

    @property
    def valence(self) -> int:
        """the maximum number of hydrogens this atom can combine with"""
        return 4 - abs(4 - self.valence_electrons - self.charge)

    @property
    def weight(self) -> float:
        """standard atomic weight"""
        return self._table.at[self.number, "weight"]

    @property
    def vdw_radius(self) -> float:
        """Returns 0.0 if not available"""
        return self._table.at[self.number, "vdw_radius"]

    @property
    def single_covalent_radius(self) -> float:
        """Returns 0.0 if not available"""
        return self._table.at[self.number, "single_covalent_radius"]

    @property
    def double_covalent_radius(self) -> float:
        """Returns 0.0 if not available"""
        return self._table.at[self.number, "double_covalent_radius"]

    @property
    def triple_covalent_radius(self) -> float:
        """Returns 0.0 if not available"""
        return self._table.at[self.number, "triple_covalent_radius"]

    @property
    def covalent_radii(self) -> Tuple[float, float, float]:
        """tuple of single, double, and triple covalent radii"""
        return self.single_covalent_radius, self.double_covalent_radius, self.triple_covalent_radius

    def copy(self) -> Atom:
        """Returns deep copy of this atom"""
        return Atom(self.number, self.x, self.y, self.z, self.charge)


class Bond():
    """
    bond in a FlexMol object.

    Attributes:
        atoms: tuple of two connected atoms
        bond_type: bond type (1: single, 2: double, 3: triple, 4: aromatic)
    """

    def __init__(self, atom0: Atom, atom1: Atom, bond_type: int = 1):
        self.atoms: Tuple[Atom, Atom] = (atom0, atom1)
        self.bond_type: int = int(bond_type)

    def other(self, atom) -> Atom:
        """Returns the other atom"""
        if self.atoms[0] == atom:
            return self.atoms[1]
        else:
            return self.atoms[0]

    @property
    def vector(self) -> np.ndarray[float]:
        """vector from atoms[0] to atoms[1]"""
        return self.atoms[1].xyz - self.atoms[0].xyz

    @property
    def length(self) -> float:
        """bond length"""
        return np.linalg.norm(self.vector)


class Lattice():
    """
    3-dimensional Bravais lattice

    Attributes:
        vectors: translational vectors (ndarray(3,3) of float)
    """

    def __init__(self, vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]):
        self.vectors = np.array(vectors[0:3][0:3])

    @property
    def ax(self):
        """
        equivalent to vectors[0][0]
        """
        return self.vectors[0][0]
    @ax.setter
    def ax(self, value):
        self.vectors[0][0] = value

    @property
    def ay(self):
        """
        equivalent to vectors[0][1]
        """
        return self.vectors[0][1]
    @ay.setter
    def ay(self, value):
        self.vectors[0][1] = value

    @property
    def az(self):
        """
        equivalent to vectors[0][2]
        """
        return self.vectors[0][2]
    @az.setter
    def az(self, value):
        self.vectors[0][2] = value

    @property
    def bx(self):
        """
        equivalent to vectors[1][0]
        """
        return self.vectors[1][0]
    @bx.setter
    def bx(self, value):
        self.vectors[1][0] = value

    @property
    def by(self):
        """
        equivalent to vectors[1][1]
        """
        return self.vectors[1][1]
    @by.setter
    def by(self, value):
        self.vectors[1][1] = value

    @property
    def bz(self):
        """
        equivalent to vectors[1][2]
        """
        return self.vectors[1][2]
    @bz.setter
    def bz(self, value):
        self.vectors[1][2] = value

    @property
    def cx(self):
        """
        equivalent to vectors[2][0]
        """
        return self.vectors[2][0]
    @cx.setter
    def cx(self, value):
        self.vectors[2][0] = value

    @property
    def cy(self):
        """
        equivalent to vectors[2][1]
        """
        return self.vectors[2][1]
    @cy.setter
    def cy(self, value):
        self.vectors[2][1] = value

    @property
    def cz(self):
        """
        equivalent to vectors[2][2]
        """
        return self.vectors[2][2]
    @cz.setter
    def cz(self, value):
        self.vectors[2][2] = value

    def len_a(self):
        """
        returns cell length a
        """
        return np.linalg.norm(self.vectors[0])

    def len_b(self):
        """
        returns cell length b
        """
        return np.linalg.norm(self.vectors[1])

    def len_c(self):
        """
        returns cell length c
        """
        return np.linalg.norm(self.vectors[2])

    def alpha(self):
        """
        returns cell angle alpha in degree
        """
        return _angle_degree(self.vectors[1], self.vectors[2])

    def beta(self):
        """
        returns cell angle beta in degree
        """
        return _angle_degree(self.vectors[2], self.vectors[0])

    def gamma(self):
        """
        returns cell angle gamma in degree
        """
        return _angle_degree(self.vectors[0], self.vectors[1])

    def cell_parameters(self):
        """
        returns [len_a, len_b, len_c, alpha, beta, gamma]
        """
        return [self.len_a(), self.len_b(), self.len_c(), self.alpha(), self.beta(), self.gamma()]

    def volume(self):
        """
        returns volume of unit cell (negative for left-handed systems)
        """
        return np.linalg.det(self.vectors)

    def copy(self):
        return Lattice(self.vectors.copy())

    def to_cartesian(self, fractional):
        """
        converts fractional coordinates into cartesian coordinates
        """
        return fractional @ self.vectors

    def to_fractional(self, cartesian):
        """
        converts cartesian coordinates into fractional coordinates
        """
        return np.linalg.solve(self.vectors.transpose(), cartesian)

    def mod_fractional(self, fractional):
        """
        mod fractional coordinates between 0.0 and 1.0
        """
        return np.mod(fractional, 1)

    def mod_cartesian(self, cartesian):
        """
        mod cartesian coordinates so that it is inside reference cell
        """
        return self.to_cartesian(np.mod(self.to_fractional(cartesian), 1))

    def set_vectors_at_random(self, minvol=1000.0, maxvol=2000.0, anisotropy=5.0):
        """
        set lattice vectors at random
        """
        self.ax = 1.0
        self.cz = self.ax * random.uniform(1.0, anisotropy)
        self.by = random.uniform(self.ax, self.cz)
        self.bx = self.ax * random.uniform(-0.5, 0.5)
        self.cx = self.ax * random.uniform(-0.5, 0.5)
        self.cy = self.by * random.uniform(-0.5, 0.5)
        self.ay = 0.0
        self.az = 0.0
        self.bz = 0.0
        self.vectors *= math.pow(random.uniform(minvol, maxvol) / self.volume(), 1.0/3.0)

    def reduced_lattice(self, eps=1e-3):
        """
        returns reduced lattice and transformation matrix by Delaunay's method
        """
        import spglib
        reduced = Lattice(vectors=spglib.delaunay_reduce(self.vectors, eps))
        mat = np.linalg.solve(reduced.vectors, self.vectors)
        for i in range(3):
            for j in range(3):
                mat[i][j] = round(mat[i][j])
        return (reduced, mat)


class FlexMol():
    """Molecule class providing flexible way of data processing

    Attributes:
        name: name of molecule
        atoms: list of atoms
        bonds: list of bonds
        lattice: lattice for crystal
        spacegroup: ID of space group
    """

    SUPPORTED_INPUT_FORMATS = ('abinit', 'acesout', 'acr', 'adfband', 'adfdftb', 'adfout', 'alc', 'aoforce', 'arc', 'axsf', 'bgf', 'box', 'bs', 'c09out', 'c3d1', 'c3d2', 'caccrt', 'can', 'car', 'castep', 'ccc', 'cdjson', 'cdx', 'cdxml', 'cif', 'ck', 'cml', 'cmlr', 'cof', 'config', 'contcar', 'contff', 'crk2d', 'crk3d', 'ct', 'cub', 'cube', 'dallog', 'dalmol', 'dat', 'dmol', 'dx', 'ent', 'exyz', 'fa', 'fasta', 'fch', 'fchk', 'fck', 'feat', 'fhiaims', 'fract', 'fs', 'fsa', 'g03', 'g09', 'g16', 'g92', 'g94', 'g98', 'gal', 'gam', 'gamess', 'gamin', 'gamout', 'got', 'gpr', 'gro', 'gukin', 'gukout', 'gzmat',
                               'hin', 'history', 'inchi', 'inp', 'ins', 'jin', 'jout', 'log', 'lpmd', 'mcdl', 'mcif', 'mdff', 'mdl', 'ml2', 'mmcif', 'mmd', 'mmod', 'mol', 'mol2', 'mold', 'molden', 'molf', 'moo', 'mop', 'mopcrt', 'mopin', 'mopout', 'mpc', 'mpo', 'mpqc', 'mrv', 'msi', 'nwo', 'orca', 'out', 'outmol', 'output', 'pc', 'pcjson', 'pcm', 'pdb', 'pdbqt', 'png', 'pos', 'poscar', 'posff', 'pqr', 'pqs', 'prep', 'pwscf', 'qcout', 'res', 'rsmi', 'rxn', 'sd', 'sdf', 'siesta', 'smi', 'smiles', 'smy', 'sy2', 't41', 'tdd', 'text', 'therm', 'tmol', 'txt', 'txyz', 'unixyz', 'vasp', 'vmol', 'xml', 'xsf', 'xtc', 'xyz', 'yob')
    SUPPORTED_OUTPUT_FORMATS = ('acesin', 'adf', 'alc', 'ascii', 'bgf', 'box', 'bs', 'c3d1', 'c3d2', 'cac', 'caccrt', 'cache', 'cacint', 'can', 'cdjson', 'cdxml', 'cht', 'cif', 'ck', 'cml', 'cmlr', 'cof', 'com', 'confabreport', 'config', 'contcar', 'contff', 'copy', 'crk2d', 'crk3d', 'csr', 'cssr', 'ct', 'cub', 'cube', 'dalmol', 'dmol', 'dx', 'ent', 'exyz', 'fa', 'fasta', 'feat', 'fh', 'fhiaims', 'fix', 'fps', 'fpt', 'fract', 'fs', 'fsa', 'gamin', 'gau', 'gjc', 'gjf', 'gpr', 'gr96', 'gro', 'gukin', 'gukout', 'gzmat', 'hin', 'inchi', 'inchikey', 'inp',
                                'jin', 'k', 'lmpdat', 'lpmd', 'mcdl', 'mcif', 'mdff', 'mdl', 'ml2', 'mmcif', 'mmd', 'mmod', 'mna', 'mol', 'mol2', 'mold', 'molden', 'molf', 'molreport', 'mop', 'mopcrt', 'mopin', 'mp', 'mpc', 'mpd', 'mpqcin', 'mrv', 'msms', 'nul', 'nw', 'orcainp', 'outmol', 'paint', 'pcjson', 'pcm', 'pdb', 'pdbqt', 'png', 'pointcloud', 'poscar', 'posff', 'pov', 'pqr', 'pqs', 'qcin', 'report', 'rinchi', 'rsmi', 'rxn', 'sd', 'sdf', 'smi', 'smiles', 'stl', 'svg', 'sy2', 'tdd', 'text', 'therm', 'tmol', 'txt', 'txyz', 'unixyz', 'vasp', 'vmol', 'xed', 'xyz', 'yob', 'zin')

    def __init__(self, mol: Any = None, format: str = "auto", kekulize = True):

        if type(mol) is FlexMol:
            self.name = mol.name
            self.atoms = mol.atoms
            self.bonds = mol.bonds
            self.lattice = mol.lattice
            self.spacegroup = mol.spacegroup
            return

        # attributes
        self.name: str = "Anonymous"
        self.atoms: List[Atom] = []
        self.bonds: List[Bond] = []
        self.lattice: Lattice = None
        self.spacegroup: int = 1

        # empty FlexMol
        if mol is None:
            return

        # check whether mol is file
        if type(mol) is str and "\n" not in mol and "\r" not in mol:
            mol_is_file = any([mol.lower().endswith("."+f)
                               for f in FlexMol.SUPPORTED_INPUT_FORMATS])
        else:
            mol_is_file = False

        # case-insensitive
        format = str(format).lower()

        # specify format automatically
        if format == "auto":
            if mol_is_file:
                format = os.path.splitext(mol)[1][1:].lower()
            elif type(mol).__name__ == "Molecule" or type(mol).__name__ == "Crystal":
                format = "csd"
            elif type(mol).__name__ == "Mol":
                format = "rd"
            elif type(mol).__name__ == "OBMol":
                format = "ob"
            elif type(mol) == str:
                format = "smiles"
            else:
                raise Exception("failed to specify format automatically")

        # read file contents if mol is file name
        if mol_is_file:
            with open(mol, "r") as f:
                mol = f.read()

        def read_any(mol: Any, format: str):

            # read MDL mol file
            if format == "mol":
                lines = mol.splitlines()
                self.name = lines[0].strip()
                natoms = int(lines[3][0:3])
                nbonds = int(lines[3][3:6])
                for i in range(natoms):
                    line = lines[4+i]
                    x = float(line[0:10])
                    y = float(line[10:20])
                    z = float(line[20:30])
                    symbol = line[31:34].strip()
                    self.add_atom(symbol, x, y, z)
                for i in range(nbonds):
                    line = lines[4+natoms+i]
                    idx0 = int(line[0:3]) - 1
                    idx1 = int(line[3:6]) - 1
                    bond_type = int(line[6:9])
                    self.add_bond(idx0, idx1, bond_type)

            # read CSD Python API object
            elif format in ("csd", "ccdc"):
                self.name = mol.identifier
                mol.assign_bond_types("unknown")
                for a in mol.atoms:
                    if a.coordinates is None:
                        self.add_atom(a.atomic_number, charge=a.formal_charge)
                    else:
                        self.add_atom(a.atomic_number,
                                      *a.coordinates, a.formal_charge)
                for b in mol.bonds:
                    if b.bond_type == 1:
                        bt = 1
                    elif b.bond_type == 2:
                        bt = 2
                    elif b.bond_type == 3:
                        bt = 3
                    elif b.bond_type in (5, 7, 9):
                        bt = 4
                    else:
                        raise Exception(
                            "bond type not compatible with FlexMol: " + b.bond_type)
                    self.add_bond(b.atoms[0].index, b.atoms[1].index, bt)

            # read RDKit Mol object
            elif format.startswith("rd"):
                read_any(rdmolfiles.MolToMolBlock(
                    Chem.AddHs(mol, addCoords=True), kekulize=kekulize), format="mol")

            # read OpenBabel OBMol object
            elif format.startswith("ob") or format == "openbabel":

                self.name = mol.GetTitle()

                # get atoms
                for a in ob.OBMolAtomIter(mol):
                    self.add_atom(a.GetAtomicNum(),
                                  a.GetX(), a.GetY(), a.GetZ(), a.GetPartialCharge())

                # get bonds
                for b in ob.OBMolBondIter(mol):
                    if b.IsAromatic():
                        self.add_bond(b.GetBeginAtomIdx()-1,
                                      b.GetEndAtomIdx()-1, 4)
                    else:
                        self.add_bond(b.GetBeginAtomIdx()-1,
                                      b.GetEndAtomIdx()-1, b.GetBondOrder())

                # get lattice and spacegroup if exists
                uc = mol.GetData(ob.UnitCell)
                if uc is not None:
                    uc = ob.toUnitCell(uc)
                    self.lattice = Lattice()
                    for i, vec in enumerate(uc.GetCellVectors()):
                        self.lattice.vectors[i, :] = vec.GetX(), vec.GetY(), vec.GetZ()
                    self.spacegroup = uc.GetSpaceGroup().GetId()

            # try OpenBabel for other formats
            elif type(mol) is str:
                obmol = ob.OBMol()
                conv = ob.OBConversion()
                if format in FlexMol.SUPPORTED_INPUT_FORMATS:
                    conv.SetInFormat(format)
                    conv.ReadString(obmol, mol)
                    read_any(obmol, format="ob")
                else:
                    raise Exception(f"unsupported format: {format}")

            else:
                raise Exception(f"unsupported format: {format}")

        read_any(mol, format)

        if self.name == "":
            self.name == "Anonymous"

    def to(self, format: str):
        """
        Convert into various formats (object or file).
        If *format* is format name, it returns an object in the format.
        If *format* is file name, it output a file in the format.

        Args:
            format (str): format name or file name
        """

        # if f is filename, get path and format
        path = None
        for f in FlexMol.SUPPORTED_OUTPUT_FORMATS:
            if format.lower().endswith("."+f):
                path = format
                format = f
                break

        # case-insensitive
        format = format.lower()

        # returns or write MDL mol file
        if format == "mol":
            lines = []
            lines.append(self.name)
            lines.append("  MATSULAB" +
                         datetime.datetime.now().strftime("%m%d%y%H%M") + "3D")
            lines.append("")
            lines.append(
                f"{len(self.atoms):>3d}{len(self.bonds):>3d}  0  0  0  0  0  0  0  0999 V2000")
            for a in self.atoms:
                lines.append(
                    f"{a.x:>10.4f}{a.y:>10.4f}{a.z:>10.4f} {a.symbol:3} 0  0  0  0  0  0  0  0  0  0  0  0")
            for b in self.bonds:
                lines.append(
                    f"{self.index(b.atoms[0])+1:>3d}{self.index(b.atoms[1])+1:>3d}{b.bond_type:>3d}  0  0  0  0")
            lines.append("M  END")
            lines.append("$$$$")
            result = "\n".join(lines + [""])

            if path is None:
                return result
            else:
                with open(path, "w", encoding="utf_8") as f:
                    f.write(result)

        elif format == "cif":

            if self.lattice is None:
                self.lattice = Lattice

            l = self.lattice

            lines = []
            lines.append(f"data_{self.name}")
            lines.append(f"_space_group_name_Hall           '{_HALL_NAMES[self.spacegroup]}'")
            lines.append(f"_space_group_name_H-M_alt        '{_HM_NAMES[self.spacegroup]}'")
            lines.append(f"_space_group_IT_number           {self.spacegroup}")
            lines.append(f"_cell_length_a                   {self.lattice.len_a():.3f}")
            lines.append(f"_cell_length_b                   {self.lattice.len_b():.3f}")
            lines.append(f"_cell_length_c                   {self.lattice.len_c():.3f}")
            lines.append(f"_cell_angle_alpha                {self.lattice.alpha():.3f}")
            lines.append(f"_cell_angle_beta                 {self.lattice.beta():.3f}")
            lines.append(f"_cell_angle_gamma                {self.lattice.gamma():.3f}")
            lines.append(f"_cell_volume                     {self.lattice.volume():.3f}")
            lines.append(f"loop_")
            lines.append(f"_atom_site_label")
            lines.append(f"_atom_site_type_symbol")
            lines.append(f"_atom_site_fract_x")
            lines.append(f"_atom_site_fract_y")
            lines.append(f"_atom_site_fract_z")
            counts = np.zeros(100, dtype=int)
            inv = np.linalg.inv(self.lattice.vectors)
            for a in self.atoms:
                counts[a.number] += 1
                frac = a.xyz[:] @ inv[:, :]
                lines.append(f"{a.symbol:>2}{counts[a.number]} {a.symbol:2} {frac[0]:.5f} {frac[1]:.5f} {frac[2]:.5f}")
            lines.append(f"loop_")
            lines.append(f"_chemical_conn_bond_atom_1")
            lines.append(f"_chemical_conn_bond_atom_2")
            lines.append(f"_chemical_conn_bond_type")
            for b in self.bonds:
                lines.append(f"{self.index(b.atoms[0]):>4} {self.index(b.atoms[1]):>4} {_BOND_TYPES[b.bond_type]}")
            result = "\n".join(lines + [""])

            if path is None:
                return result
            else:
                with open(path, "w", encoding="utf_8") as f:
                    f.write(result)

        # returns SMILES
        elif format == "smiles":
            obmol = self.to("ob")
            conv = ob.OBConversion()
            conv.SetOutFormat(format)
            return conv.WriteString(obmol).split()[0]

        # returns OpenBabel object
        elif format.startswith("ob") or format == "openbabel":
            result = ob.OBMol()
            conv = ob.OBConversion()
            conv.SetInFormat("mol")
            conv.ReadString(result, self.to("mol"))
            return result

        # returns RDKit object
        elif format.startswith("rd"):
            molblock = self.to("mol")
            rdmol = rdmolfiles.MolFromMolBlock(molblock, removeHs=False)
            if rdmol is None:
                raise Exception(
                    f"RDKit failed to read mol block below:\n{molblock}")
            else:
                return rdmol

        # returns CSD Python API object
        elif format in ("csd", "ccdc"):
            return io.MoleculeReader(self.to("mol"))[0]

        # output 2D diagram
        elif format in ("png", "svg"):
            if path is None:
                raise Exception(f"Conversion to {format} requires file name.")
            else:
                subprocess.run(["obabel", "-imol", "-O", path, "-d", "--gen2d"],
                               input=self.to("mol"), encoding="utf_8", capture_output=True)

        # try OpenBabel for other formats
        else:
            obmol = ob.OBMol()
            conv = ob.OBConversion()
            if format in FlexMol.SUPPORTED_OUTPUT_FORMATS:
                conv.SetInAndOutFormats("mol", format)
                conv.ReadString(obmol, self.to("mol"))
                if path is None:
                    return conv.WriteString(obmol)
                else:
                    conv.WriteFile(obmol, path)
            else:
                raise Exception(f"unsupported format: {format}")

    def _as_atom(self, atom) -> Atom:
        """Convert atom into Atom object"""
        if isinstance(atom, Atom):
            return atom
        elif isinstance(atom, int):
            return self.atoms[atom]
        else:
            raise Exception("cannot interpret as Atom: " + str(atom))

    def _as_bond(self, bond) -> Bond:
        """Convert bond int Bond object"""
        if isinstance(bond, Bond):
            return bond
        elif isinstance(bond, int):
            return self.bonds[bond]
        elif isinstance(bond, Sequence):
            bond = map(self._as_atom, bond)
            return self.bond_between(*bond)

    def index(self, atom_or_bond) -> int:
        """Returns the index of atom or bond object. Indices start with 0."""
        if isinstance(atom_or_bond, abc.Sequence):
            return map(self.index, atom_or_bond)

        if isinstance(atom_or_bond, Atom):
            return self.atoms.index(atom_or_bond)
        elif isinstance(atom_or_bond, Bond):
            return self.bonds.index(self._as_bond(atom_or_bond))
        else:
            raise Exception("Only Atom or Bond objects are acceptable.")

    def neighbors_of(self, atom: Atom) -> List[Atom]:
        """Returns list of neighboring atoms of *atom*"""
        atom = self._as_atom(atom)
        return [b.other(atom) for b in self.bonds if atom in b.atoms]

    def bonds_of(self, atom: Atom) -> List[Bond]:
        """Returns list of bonds of *atom*"""
        atom = self._as_atom(atom)
        return [b for b in self.bonds if atom in b.atoms]

    def bond_between(self, atom0: Atom, atom1: Atom) -> Bond:
        """Returns the bond between *atom0* and *atom1*. It returns None if *atom0* is not bonded with *atom1*."""
        atoms = set(map(self._as_atom, (atom0, atom1)))
        for b in self.bonds:
            if atoms == set(b.atoms):
                return b
        return None

    def add_atom(self, number: int, x: float = 0.0, y: float = 0.0, z: float = 0.0, charge: int = 0) -> Atom:
        atom = Atom(number, x, y, z, charge)
        self.atoms.append(atom)
        return atom

    def add_bond(self, atom0: Atom, atom1: Atom, bond_type: int = 1) -> Bond:
        if type(atom0) is int:
            atom0 = self.atoms[atom0]
        if type(atom1) is int:
            atom1 = self.atoms[atom1]
        bond = Bond(atom0, atom1, bond_type)
        self.bonds.append(bond)
        return bond

    def add_mol(self, mol: FlexMol):
        """Add the copies of all atoms and bonds of another object to this object"""
        if isinstance(mol, FlexMol):
            mol = mol.copy()
        else:
            mol = FlexMol(mol)
        self.atoms += mol.atoms
        self.bonds += mol.bonds

    def remove_atom(self, atom: Atom) -> Atom:
        """Remove atom and related bonds from FlexMol. *atom* may be index of the atom."""
        atom = self._as_atom(atom)
        if atom in self.atoms:
            for b in self.bonds_of(atom):
                self.remove_bond(b)
            self.atoms.remove(atom)
            return atom
        else:
            return None

    def remove_bond(self, bond: Bond) -> Bond:
        """Remove bond from FlexMol. *bond* can be Bond, int, (Atom, Atom), or (int, int)."""
        bond = self._as_bond(bond)
        if bond in self.bonds:
            self.bonds.remove(bond)
            return bond
        else:
            return None

    def copy(self) -> FlexMol:
        """Returns deep copy of this object"""
        mol = FlexMol()
        mol.name = self.name
        for a in self.atoms:
            mol.add_atom(a.number, a.x, a.y, a.z, a.charge)
        for b in self.bonds:
            mol.add_bond(*self.index(b.atoms), b.bond_type)
        if self.lattice is not None:
            mol.lattice = self.lattice.copy()
        mol.spacegroup = self.spacegroup
        return mol

    def split(self) -> List[FlexMol]:
        """Split into isolated molecules"""

        self = self.copy()

        label = np.full(len(self.atoms), -1)
        nmols = -1

        def make_neighbors_same(atom: Atom):
            for neighbor in self.neighbors_of(atom):
                idx = self.index(neighbor)
                if label[idx] == -1:
                    label[idx] = nmols
                    make_neighbors_same(self.atoms[idx])

        for i, a in enumerate(self.atoms):
            if label[i] == -1:
                nmols += 1
                label[i] = nmols
                make_neighbors_same(a)

        mols = []
        for i in range(nmols + 1):
            mol = FlexMol()
            mol.name = self.name
            # mol.lattice[:, :] = self.lattice[:, :]
            for l, a in zip(label, self.atoms):
                if l == i:
                    mol.atoms.append(a)
            for b in self.bonds:
                if label[self.index(b.atoms[0])] == i:
                    mol.bonds.append(b)
            for a in mol.atoms:
                a.mol = mol
            for b in mol.bonds:
                b.mol = mol
            mols.append(mol)
        return mols

    def num_hydrogens(self, atom: Atom) -> int:
        """Returns the number of hydrogens attached to *atom*"""
        return sum([a.number == 1 for a in self.neighbors_of(atom)])

    def detach_hydrogens(self, atom: Atom, num: int = 1) -> List[Atom]:
        atom = self._as_atom(atom)
        removed = []
        for a in self.neighbors_of(atom):
            if num == 0:
                break
            if a.number == 1:
                removed.append(self.remove_atom(a))
                num -= 1
        return removed

    def attach_atom(self, target: Atom, element: int, bond_type: int = 1, del_hydrogens: int = "auto", add_hydrogens: int = "auto") -> Atom:
        """Attach a new atom of *element* to the *target* atom. Hydrogens are attached to the new atom if required."""
        target = self._as_atom(target)

        # delete hydrogens of the target atom
        if del_hydrogens == "auto":
            del_hydrogens = bond_type
        if del_hydrogens is None or del_hydrogens == False:
            del_hydrogens = 0
        if del_hydrogens > 0:
            self.detach_hydrogens(target, del_hydrogens)

        # add new atom and bond
        new_atom = self.add_atom(element, target.x, target.y, target.z)
        self.add_bond(target, new_atom, bond_type)

        # add hydrogens if required
        if add_hydrogens == "auto":
            add_hydrogens = new_atom.valence - bond_type
        if add_hydrogens is None or add_hydrogens == False:
            add_hydrogens = 0
        for _ in range(add_hydrogens):
            self.attach_atom(new_atom, 1, del_hydrogens=0, add_hydrogens=0)

    def add_hydrogens(self):
        """Hydrogens are added using RDKit."""
        # mol = FlexMol(Chem.AddHs(self.to("rd")))
        obmol = self.to("ob")
        obmol.AddHydrogens()
        new_mol = FlexMol(obmol)
        self.atoms = new_mol.atoms
        self.bonds = new_mol.bonds

    def gen3d(self, kekulize=True):
        """Generate 3D coordinates using RDKit. Hydrogens are added if needed."""
        rdmol = self.to("rd")
        rdmol = Chem.AddHs(rdmol)
        AllChem.EmbedMolecule(rdmol)
        AllChem.MMFFOptimizeMolecule(rdmol)
        mol = FlexMol(rdmol, kekulize=kekulize)
        self.atoms = mol.atoms
        self.bonds = mol.bonds

    def center(self, weights=None):
        """Returns weighted center. *weights=None* returns geometrical center. *weights="mass"* returns mass-weighted center."""
        if isinstance(weights, str) and weights.lower().startswith("geom"):
            weights = None
        if weights == "mass":
            weights = [a.weight for a in self.atoms]
        return np.average([a.xyz for a in self.atoms], axis=0, weights=weights)

    def translate(self, vector):
        """Translate all atoms by the vector"""
        for a in self.atoms:
            a.xyz += vector

    def rotate(self, axis: np.ndarray, angle: float, center=(0.0, 0.0, 0.0)):
        """Rotate molecule around the axis at the center by the angle (radian). *center="geometry"* uses geometrical center, and *center="mass"* uses mass center."""
        axis *= angle / np.linalg.norm(axis)
        rot = Rotation.from_rotvec(axis)
        if isinstance(center, str) and center.lower().startswith("geom"):
            center = self.center()
        elif center == "mass":
            center = self.center(weights="mass")
        tra = np.array(center)
        for a in self.atoms:
            a.xyz[:] = rot.apply(a.xyz - tra) + tra

    def randomize(self, amplitude: float = 0.01):
        for a in self.atoms:
            a.x += random.uniform(-amplitude, amplitude)
            a.y += random.uniform(-amplitude, amplitude)
            a.z += random.uniform(-amplitude, amplitude)


if __name__ == "__main__":

    count = 0
    for mol in io.MoleculeReader("CSD"):
        FlexMol(mol).to(f"{count}.png")
        count += 1
        if count > 10:
            break

    benzene = FlexMol("c1ccccc1")
    benzene.name = "benzene"
    benzene.to("benzene.png")

    benzene.gen3d()
    benzene.to("benzene1.mol")
    rdmol = benzene.to("rd")
    FlexMol(rdmol).to("benzene2.mol")
    csdmol = FlexMol(rdmol).to("csd")
    FlexMol(csdmol).to("benzene3.mol")
    obmol = FlexMol(csdmol).to("ob")
    FlexMol(obmol).to("benzene4.mol")
    print(FlexMol(obmol).to("smiles"))
