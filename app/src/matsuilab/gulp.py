"""
Python interface for GULP + Free energy minimization
"""

import math
import subprocess
import sys
from typing import Any, Sequence

import numpy as np
import numpy.linalg as LA

from matsuilab import atpro


class Job():
    """
    GULP job class.
    Either *cell* or *vectors* should be given.
    Either *fractional* or *cartesian* should be given.
    """

    def __init__(self, header: str = None,
        cell: np.ndarray = None, vectors: np.ndarray = None,
        symbols: Sequence[str] = None, fractional: np.ndarray = None,
        cartesian: np.ndarray = None, charges: np.ndarray = None,
        result: str = None):

        self.header = header
        self.symbols = symbols
        self._fractional = fractional
        self._cartesian = cartesian
        self.charges = charges
        self.cell = cell
        self._vectors = vectors
        self.result = result

    def to_gin(self):
        """Returns content of GULP input (gin) file"""

        lines = [self.header]

        if self.cell is not None:
            lines.append(
                "\ncell\n" + " ".join([f"{c:.9}" for c in self.cell]) + " 0 0 0 0 0 0")

        if self.vectors is not None:
            lines.append("\nvectors")
            for row in self.vectors:
                lines.append(" ".join([f"{v:.9}" for v in row]))
            lines.append("0 0 0 0 0 0")

        if self.fractional is not None:
            lines.append("\nfractional")
            if self.charges is None:
                for s, p in zip(self.symbols, self.fractional):
                    lines.append(f"{s:5} {p[0]:.9} {p[1]:.9} {p[2]:.9} 1 1 1")
            else:
                for s, p, c in zip(self.symbols, self.fractional, self.charges):
                    lines.append(
                        f"{s:5} {p[0]:.9} {p[1]:.9} {p[2]:.9} {c:.9} 1 1 1")

        if self.cartesian is not None:
            lines.append("\ncartesian")
            if self.charges is None:
                for s, p in zip(self.symbols, self.cartesian):
                    lines.append(f"{s:5} {p[0]:.9} {p[1]:.9} {p[2]:.9} 1 1 1")
            else:
                for s, p, c in zip(self.symbols, self.cartesian, self.charges):
                    lines.append(
                        f"{s:5} {p[0]:.9} {p[1]:.9} {p[2]:.9} {c:.9} 1 1 1")

        return "\n".join(lines)

    def run(self, timeout: float = None):
        """
        Executes GULP

        Args:
            timeout: in second
        
        Raises:
            Exception: error termination of GULP
            TimeoutExpired: timeout
        """
        self.result = subprocess.run("gulp", input=self.to_gin(), text=True, capture_output=True, timeout=timeout).stdout
        if self.error_message is not "":
            raise Exception("Error Termination in GULP!\n" + self.error_message)
        elif self.error_termination:
            raise Exception("Error Termination in GULP!")

    @property
    def error_termination(self) -> bool:
        for line in self.result.splitlines():
            if line.startswith("  Job Finished at "):
                return False
        return True

    @property
    def error_message(self) -> str:
        return "\n".join([line for line in self.result.splitlines() if line.startswith("!!")])

    @property
    def energy(self) -> float:
        e = None
        for line in self.result.splitlines():
            if line.startswith("  Total ") and line.endswith("eV"):
                e = line.split()[-2]
        if e is None:
            return None
        else:
            return float(e)

    @property
    def vectors(self) -> np.ndarray:
        if self.result is None:
            return self._vectors
        else:
            vectors = np.zeros((3, 3))
            lines = self.result.splitlines()
            start = lines.index("  Final Cartesian lattice vectors (Angstroms) :")
            for i in range(3):
                vectors[i, :] = [float(v)
                                for v in lines[start+2+i].strip().split()]
            return vectors

    @vectors.setter
    def vectors(self, value):
        self._vectors = value

    @property
    def fractional(self) -> np.ndarray:
        if self.result is None:
            return self._fractional
        else:
            lines = self.result.splitlines()

            for line in lines:
                if line.startswith("  Total number atoms/shells = "):
                    n = int(line.split()[4])
                    positions = np.zeros((n, 3))
                    break

            start = lines.index("  Final fractional coordinates of atoms :")
            for i in range(n):
                positions[i, :] = [float(v)
                                for v in lines[start+6+i].strip().split()[3:6]]
            return positions

    @fractional.setter
    def fractional(self, value):
        self._fractional = value

    @property
    def cartesian(self) -> np.ndarray:
        if self.result is None:
            return self._cartesian
        else:
            lines = self.result.splitlines()

            for line in lines:
                if line.startswith("  Total number atoms/shells = "):
                    n = int(line.split()[4])
                    positions = np.zeros((n, 3))
                    break

            start = lines.index("  Final cartesian coordinates of atoms :")
            for i in range(n):
                positions[i, :] = [float(v)
                                for v in lines[start+6+i].strip().split()[3:6]]
            return positions

    @cartesian.setter
    def cartesian(self, value):
        self._cartesian = value


def minimize_free_energy(header1: str = None, header2: str = None,
    cell: np.ndarray = None, vectors: np.ndarray = None,
    symbols: Sequence[str] = None, fractional: np.ndarray = None,
    cartesian: np.ndarray = None, charges: np.ndarray = None,
    max_cyc: int = 100):
    """
    Performs free energy minimization and returns optimized coodinates as a tuple of fractional and vectors.
    *header1* should include *optimise* keyword, and exclude *free* and *conp* keywords.
    *header2* should include *free* keyword, and exclude *optimise* and *conp* keywords. *temperature* option can be included.
    
    Args:
        header1: for internal optimization.
        header2: for free energy calculation.
    """

    if cell is not None and vectors is None:
        vectors = _cell_to_vectors(cell[0:3], cell[3:6])
    if fractional is None and cartesian is not None:
        fractional = LA.solve(vectors, cartesian)

    delta = 1e-2
    alpha = 1e0
    min_alpha = 1e-2
    max_alpha = 1e1

    def opt_frac(vec=None, frac=None):
        if vec is None:
            vec = vectors
        if frac is None:
            frac = fractional
        job = Job(header=header1, symbols=symbols, fractional=frac, charges=charges, vectors=vec)
        job.run()
        return job.fractional

    def calc_free_energy(vec=None, frac=None):
        if vec is None:
            vec = vectors
        if frac is None:
            frac = opt_frac(vec=vec)
        job = Job(header=header2, symbols=symbols, fractional=frac, charges=charges, vectors=vec)
        job.run()
        return job.energy

    def calc_e0_deriv():

        # update fractional and calculate energy
        fractional[:, :] = opt_frac()
        e0 = calc_free_energy()

        # update deriv
        deriv = np.zeros((3, 3))
        vec = vectors.copy()
        for indices in ((0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)):
            vec[indices] = vectors[indices] + delta
            deriv[indices] = (calc_free_energy(vec=vec) - e0) / delta
            vec[indices] = vectors[indices]
        
        return e0, deriv

    print(f"Cycle     Free_Energy     Cell_Volume           Gnorm")

    success = False

    for cyc in range(max_cyc):

        e0, deriv = calc_e0_deriv()

        # print current status
        print(f"{cyc:5} {e0:15.9} {abs(LA.det(vectors)):15.9} {LA.norm(deriv):15.9}")

        # previous energy
        e2 = sys.float_info.max

        if alpha < min_alpha:
            alpha = min_alpha
        if alpha > max_alpha:
            alpha = max_alpha

        # line search
        for _ in range(10):
            e1 = calc_free_energy(vec=vectors - deriv * (delta * alpha / LA.norm(deriv)))
            if e1 < e0: # better than original. Try with larger alpha.
                if alpha * 2 > max_alpha: # alpha exceeds upper limit
                    break
                else:
                    alpha *= 2.0
                    e2 = e1
            elif e2 < e0: # previous value was good
                alpha *= 0.5
                break
            else: # not good. Try with smaller alpha.
                if alpha * 0.5 < min_alpha: # alpha is below lower limit
                    break
                else:
                    alpha *= 0.5
                    e2 = e1

        if e1 < e0: # if improved, update vectors
            vectors[:, :] -= deriv * (delta * alpha / LA.norm(deriv))
        else: # if no better value found, terminate
            success = True
            break

    e0, deriv = calc_e0_deriv()

    # print current status
    print(f"Final {e0:15.9} {abs(LA.det(vectors)):15.9} {LA.norm(deriv):15.9}")
    if success:
        print("\nOptimization achieved.")
    else:
        print("\nOptimization failed.")

    return fractional, vectors


def _cell_to_vectors(lengths, angles):
    """convert cell lengths and angles to lattice vectors"""

    cos_alpha = math.cos(np.deg2rad(angles[0]))
    cos_beta = math.cos(np.deg2rad(angles[1]))
    cos_gamma = math.cos(np.deg2rad(angles[2]))
    sin_gamma = math.sin(np.deg2rad(angles[2]))

    vectors = np.zeros((3, 3))
    vectors[0, 0] = lengths[0]
    vectors[1, 0] = lengths[1] * cos_gamma
    vectors[1, 1] = lengths[1] * sin_gamma
    vectors[2, 0] = lengths[2] * cos_beta
    vectors[2, 1] = lengths[2] * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    vectors[2, 2] = lengths[2] * \
        math.sqrt(1.0 - cos_beta**2 -
                  ((cos_alpha - cos_beta * cos_gamma) / sin_gamma)**2)

    return vectors


def _main():
    import ccdc

    cry = ccdc.io.CrystalReader("CSD").crystal("BENZEN03")
    mol = cry.packing(box_dimensions=((0, 0, 0), (0.999, 0.999, 0.999)))

    cell = list(cry.cell_lengths) + list(cry.cell_angles)
    positions = np.array([a.fractional_coordinates for a in mol.atoms])
    numbers = np.array([a.atomic_number for a in mol.atoms], dtype=int)
    symbols = [atpro.SYMBOLS[n] for n in numbers]

    charges = np.zeros(len(numbers))
    for i, n in enumerate(numbers):
        if n == 1:
            charges[i] = 0.109
        else:
            charges[i] = -0.109

    header1 = "molecule c6 optimise\n\n" \
        "library dreiding\n\n" \
        "species\n" \
        "C C_R\n" \
        "H H_"
    header2 = "molecule c6 free\n\n" \
        "library dreiding\n" \
        "temperature 218 K\n\n" \
        "species\n" \
        "C C_R\n" \
        "H H_"
    minimize_free_energy(header1=header1, header2=header2, symbols=symbols, fractional=positions, charges=charges, cell=cell)


if __name__ == "__main__":
    _main()
