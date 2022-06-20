import math
import os
import re
import subprocess
from typing import List, Sequence, Tuple, Union

import numpy as np

from matsuilab import atpro, const, unit
from matsuilab.chem import FlexMol

gaussian_command = "g16"


def main():

    cond = {"link0": "nprocshared=2",
            "route": "opt b3lyp/6-31g scrf=(cpcm,read)",
            "optional": "Eps=38"}

    job = GaussianJobFile(**cond)
    job2 = GaussianJobFile(**cond)
    job.append(job2)
    log = job.run("test.gjf")
    print(log.normal_termination())
    # gjf.join_write(gjf, "test.gjf")


def _split(pattern, string):

    if type(string) is str:
        result = re.split(pattern, string.strip())
    else:
        result = sum([re.split(pattern, s.strip()) for s in string], [])

    # remove empty elements
    return [r for r in result if r != ""]


def _delete_file(file):
    if os.path.exists(file):
        os.remove(file)
        return True
    else:
        False


def _rename_file(old_file, new_file):
    if os.path.exists(old_file):
        _delete_file(new_file)
        os.rename(old_file, new_file)
        return True
    else:
        return False


class GaussianJobFile:
    """
    Attributes:
        link0: Link 0 commands such as "%nprocshared=9"
        plevel: print level ("n": normal, "p": precise, "t": terse)
        route: route section such as "opt b3lyp/6-31g(d)"
        title: title (up to 5 lines)
        charge: total charge
        spin: spin multiplicity
        mol: molecule specification in any formats supported by FlexMol
        optional: optional sections such as connectivity
    """

    def __init__(self,
                 link0: Union[str, Sequence[str]] = "",
                 plevel: str = "p",
                 route: Union[str, Sequence[str]] = "opt b3lyp/6-31g(d)",
                 title: str = "Anonymous",
                 charge: int = 0,
                 spin: int = None,
                 mol: FlexMol = None,
                 optional: Union[str, Sequence[str]] = ""):

        self.link0 = _split("[%\r\n]", link0)
        self.plevel = plevel
        self.route = _split("[#\r\n ]", route)
        self.title = title.strip()
        self.charge = int(charge)
        self.spin = spin
        self.mol = FlexMol(mol)
        self.optional = _split("\r\n", optional)
        self.link1 = []

    def __str__(self):
        atom_list = []
        if self.mol is not None:
            for atom in self.mol.atoms:
                atom_list.append(f"{atom.symbol:2} {atom.x:14.8f} {atom.y:14.8f} {atom.z:14.8f}")
        if self.spin is None:
            self.spin = 1 + sum([atom.number - atom.charge for atom in self.mol.atoms]) % 2
        return "\n".join(
            [f"%{l}" for l in self.link0] +
            [f"#{self.plevel} " + " ".join(self.route)] +
            [""] +
            [self.title] +
            [""] +
            [f"{self.charge} {self.spin}"] +
            atom_list +
            [""] +
            self.optional +
            [""] +
            [""] +
            ["--Link1--\n" + str(l) for l in self.link1]
        )

    def write(self, path):
        with open(path, "w", encoding="utf_8") as f:
            f.write(str(self))

    def append(self, subjob):
        self.link1.append(subjob)

    # def join(self, subfiles):
    #     if isinstance(subfiles, GaussianJobFile):
    #         subfiles = [subfiles]
    #     return "--Link1--\n".join(map(str, [self] + subfiles))

    # def join_write(self, subfiles, path):
    #     with open(path, "w", encoding="utf_8") as f:
    #         f.write(self.join(subfiles))

    def run(self, path):
        self.write(path)
        subprocess.run([gaussian_command, path])
        basename = os.path.splitext(path)[0]
        _rename_file(basename+".out", basename+".log")
        return GaussianLogFile(basename+".log")


class GaussianLogFile():
    """An object to read a variety of information from a Gaussian log file (.log)"""

    def __init__(self, path: str):
        self.path = path
        self.filename = os.path.basename(path)
        self.lines = None
        with open(path, "r") as f:
            self.lines = f.readlines()

    def normal_termination(self) -> bool:
        """Returns whether all jobs terminated normally"""
        if self.lines is None:
            return False
        else:
            count1 = 0
            count2 = 0
            for l in self.lines:
                if l.startswith(" Entering Link 1"):
                    count1 += 1
                if l.startswith(" Normal termination"):
                    count2 += 1
            return count1 == count2

    def final_energy(self, unit: str = "eV") -> float:
        """Returns final energy"""
        energy = None
        for l in self.lines:
            if l.startswith(" SCF Done:"):
                energy = float(l.split()[4])
        if unit == "eV":
            energy *= 27.2114
        return energy

    def homo_lumo_energies(self, unit: str = "eV") -> Tuple[float, float]:
        """Returns HOMO and LUMO energies as tuple"""
        homo = None
        lumo = None
        flag = False  # 空軌道の１行目だけを読むために使用
        for line in self.lines:
            # 被占軌道の最後の固有エネルギーを取得
            if line.startswith(" Alpha  occ. eigenvalues -- "):
                homo = line[28:].split()[-1]
                flag = True
            # 空軌道の最初の固有エネルギーを取得
            if flag and line.startswith(" Alpha virt. eigenvalues -- "):
                lumo = line[28:].split()[0]
                flag = False
        if homo is not None:
            homo = float(homo)
            if unit == "eV":
                homo *= 27.2114
        if lumo is not None:
            lumo = float(lumo)
            if unit == "eV":
                lumo *= 27.2114
        return homo, lumo

    def l9999_output(self) -> List[str]:
        """Returns output of l9999.exe"""
        found = False
        buffer = []
        for line in self.lines:
            if line.strip().endswith("l9999.exe)"):
                found = True
                buffer.clear()
            elif found:
                if line.strip() == "":
                    found = False
                else:
                    buffer.append(line[1:-1])

        join_buffer = "".join(buffer)
        if join_buffer.count("|") > join_buffer.count("\\"):
            return join_buffer.split("|")  # windows
        else:
            return join_buffer.split("\\")  # linux

    def atoms(self) -> List[Tuple[str, float, float, float]]:
        """Returns list of element symbols and final x, y, z coordinates"""
        buffer = self.l9999_output()
        blank_index = [i for i, b in enumerate(buffer) if b == ""]
        atoms = []
        for b in buffer[blank_index[2]+2:blank_index[3]]:
            a = b.split(",")
            if len(a) == 4:
                atoms.append((a[0], float(a[1]), float(a[2]), float(a[3])))
            elif len(a) == 5:
                atoms.append((a[0], float(a[2]), float(a[3]), float(a[4])))
            else:
                raise NotImplementedError(a)

        return atoms

    def atomic_numbers(self) -> np.ndarray:
        """Returns 1D array of atomic numbers"""
        buffer = self.l9999_output()
        blank_index = [i for i, b in enumerate(buffer) if b == ""]
        buffer = buffer[blank_index[2]+2:blank_index[3]]
        numbers = np.zeros(len(buffer))
        for i, b in enumerate(buffer):
            numbers[i] = atpro.SYMBOLS.index(b.split(",")[0])
        return numbers

    def coordinates(self) -> np.ndarray:
        """Returns 2D array of x, y, z coordinates of atoms"""
        buffer = self.l9999_output()
        blank_index = [i for i, b in enumerate(buffer) if b == ""]
        buffer = buffer[blank_index[2]+2:blank_index[3]]
        coord = np.zeros((len(buffer), 3))
        for c, b in zip(coord, buffer):
            a = b.split(",")
            if len(a) == 4:
                c[:] = float(a[1]), float(a[2]), float(a[3])
            elif len(a) == 5:
                c[:] = float(a[2]), float(a[3]), float(a[4])
            else:
                raise NotImplementedError(a)
        return coord

    def job_cpu_time(self) -> float:
        """Returns sum of cpu time in seconds"""
        t = 0.0
        for l in self.lines:
            if l.startswith(" Job cpu time:"):
                split = l.split()
                day, hour, min, sec = float(split[3]), float(
                    split[5]), float(split[7]), float(split[9])
                t += ((day * 24 + hour) * 60 + min) * 60 + sec
        return t

    def elapsed_time(self) -> float:
        """Returns sum of elapsed time in seconds"""
        t = 0.0
        for l in self.lines:
            if l.startswith(" Elapsed time:"):
                split = l.split()
                day, hour, min, sec = float(split[2]), float(
                    split[4]), float(split[6]), float(split[8])
                t += ((day * 24 + hour) * 60 + min) * 60 + sec
        return t


class GaussianCubeFile():
    """An object to read a Gaussian cube file (.cube)

    Attributes:
        self.path (str): file path
        self.header (str): header information
        natoms (int): the number of atoms
        origin (np.ndarray): coordinates of origin in Angstrom
        nval (int): the number of values per point (usually nval=1)
        npts ((int, int, int)): the number of mesh points
        mesh (np.ndarray((3, 3))): translational vectors of mesh points in Angstrom
        atom_num (np.ndrray): atomic numbers
        charges (np.ndarray): charges of respective atoms
        coordinates (np.ndarray): cartesian coordinates fo atoms in Angstrom
        array (np.ndarray): values at respective mesh points
    """

    def __init__(self, path: str):
        self.path = path
        self.header = ""
        with open(path, "r") as f:

            # read header if exists
            for _ in range(10):
                line = f.readline()
                if not all([(c in " -1234567890.\r\n") for c in line]):
                    self.header += line
                else:
                    break

            # read NAtoms, X-Origin, Y-Origin, Z-Origin, NVal
            tokens = line.strip().split()
            self.natoms = int(tokens[0])
            self.origin = np.array([float(t) for t in tokens[1:4]])
            self.nval = int(tokens[4])
            self.origin *= const.bohr_radius / unit.Angstrom

            # read mesh information
            self.npts = np.empty(3, dtype=int)
            self.mesh = np.empty((3, 3))
            for i in range(3):
                tokens = f.readline().strip().split()
                self.npts[i] = int(tokens[0])
                self.mesh[i, :] = [float(t) for t in tokens[1:4]]
            self.mesh *= const.bohr_radius / unit.Angstrom

            # read atom list
            self.atom_num = np.empty(self.natoms, dtype=int)
            self.charges = np.empty(self.natoms)
            self.coordinates = np.empty((self.natoms, 3))
            for i in range(self.natoms):
                tokens = f.readline().strip().split()
                self.atom_num[i] = int(tokens[0])
                self.charges[i] = float(tokens[1])
                self.coordinates[i, :] = [float(t) for t in tokens[2:5]]
            self.coordinates *= const.bohr_radius / unit.Angstrom

            # read values of all points
            if self.nval == 1:
                self.array = np.empty(self.npts)
            else:
                self.array = np.empty((*self.npts, self.nval))
            flat = self.array.reshape(-1)
            idx = 0
            while idx < flat.size:
                tokens = f.readline().strip().split()
                flat[idx:idx+len(tokens)] = [float(t) for t in tokens]
                idx += len(tokens)

    def mask(self, offset=0.0, outside=False):
        """returns mask array which has True inside van der Waals radii (+offset)."""
        from openbabel import openbabel as ob

        dx = self.mesh[0, 0]
        dy = self.mesh[1, 1]
        dz = self.mesh[2, 2]

        # calculate the cartesian coordinates of all mesh points
        mesh_coordinates = np.empty((*self.npts, 3))
        # for i in range(self.npts[0]):
        #     mesh_coordinates[i, :, :, 0] = self.origin[0] + dx * i
        # for i in range(self.npts[1]):
        #     mesh_coordinates[:, i, :, 1] = self.origin[1] + dy * i
        # for i in range(self.npts[2]):
        #     mesh_coordinates[:, :, i, 2] = self.origin[2] + dz * i
        for i in range(self.npts[0]):
            mesh_coordinates[i, :, :, 0] = i
        for i in range(self.npts[1]):
            mesh_coordinates[:, i, :, 1] = i
        for i in range(self.npts[2]):
            mesh_coordinates[:, :, i, 2] = i
        mesh_coordinates = self.forward(mesh_coordinates)

        # make mask True for all points inside van der Waals radii
        mask = np.zeros(self.array.shape, dtype=bool)
        for num, coord in zip(self.atom_num, self.coordinates):
            r = ob.GetVdwRad(int(num)) + offset
            # define box which covers the sphere of r
            x0 = max(math.floor((coord[0] - r - self.origin[0]) / dx), 0)
            x1 = min(
                math.ceil((coord[0] + r - self.origin[0]) / dx) + 1, self.npts[0])
            y0 = max(math.floor((coord[1] - r - self.origin[1]) / dy), 0)
            y1 = min(
                math.ceil((coord[1] + r - self.origin[1]) / dy) + 1, self.npts[1])
            z0 = max(math.floor((coord[2] - r - self.origin[2]) / dz), 0)
            z1 = min(
                math.ceil((coord[2] + r - self.origin[2]) / dz) + 1, self.npts[2])
            mask[x0:x1, y0:y1, z0:z1] = np.logical_or(mask[x0:x1, y0:y1, z0:z1], np.linalg.norm(
                mesh_coordinates[x0:x1, y0:y1, z0:z1, :] - coord[:], axis=3) <= r)

        if outside:
            mask = np.logical_not(mask)
        return mask

    def masked_array(self, offset=0.0, outside=False):
        """returns array where inside van der Waals radii (+offset) is masked"""
        return np.ma.masked_array(self.array, mask=self.mask(offset=offset, outside=outside))

    def forward(self, indices):
        """convert mesh indices to cartesian coordinates"""
        return indices @ self.mesh + self.origin

    def backward(self, coordinates):
        """convert cartesian coordinates to mesh indices"""
        return (coordinates - self.origin) @ np.linalg.inv(self.mesh)

    def max_cartesian(self, mask=True, offset=0.0, outside=False):
        """returns the position where the cube value has maximum"""
        if mask:
            masked_array = self.masked_array(offset=offset, outside=outside)
            return self.forward(np.unravel_index(masked_array.argmax(), masked_array.shape))
        else:
            return self.forward(np.unravel_index(np.argmax(self.array), self.array.shape))

    def min_cartesian(self, mask=True, offset=0.0, outside=False):
        """returns the position where the cube value has minumum"""
        if mask:
            masked_array = self.masked_array(offset=offset, outside=outside)
            return self.forward(np.unravel_index(masked_array.argmin(), masked_array.shape))
        else:
            return self.forward(np.unravel_index(np.argmax(self.array), self.array.shape))


def formchk():
    pass


def cubegen(fchkfile, cubefile, nproc=1, kind="Potential=SCF", npts=-2, format="h", cubefile2=None):
    if cubefile2 is None:
        subprocess.run(["cubegen", str(nproc), kind, fchkfile, cubefile, str(npts), format])
    elif type(cubefile2) is str:
        subprocess.run(["cubegen", str(nproc), kind, fchkfile, cubefile, str(npts), format, cubefile2])
    elif type(cubefile2) is tuple:
        origin = cubefile2[0]
        npoints = cubefile2[1]
        vectors = cubefile2[2]
        cube2 =  f"   -1 {origin[0]:11.6f} {origin[1]:11.6f} {origin[2]:11.6f}    1\n"
        for i in range(3):
            cube2 += f"{npoints[i]:5d} {vectors[i][0]:11.6f} {vectors[i][1]:11.6f} {vectors[i][2]:11.6f}\n"
        subprocess.run(["cubegen", str(nproc), kind, fchkfile, cubefile, str(npts), format],
        input=cube2, text=True)
    else:
        raise Exception(
            "cubefile2 must be None, str, or tuple of (origin, npts, vectors)")


if __name__ == "__main__":
    main()
