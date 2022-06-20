from matsuilab import chem, gulp, atpro
import numpy as np
import numpy.linalg as LA
import random
import math
import itertools
import statistics
from concurrent.futures import ProcessPoolExecutor
import time
import datetime
import sys

def generate_initial_crystals():
    pass


def gen3d(smiles):
    mol = chem.FlexMol(smiles)
    mol.gen3d()
    return mol


def optimize_conformer(mol):
    # symbols and cartesian
    symbols = []
    for a in mol.atoms:
        if a.number == 1:
            symbols.append("H")
        elif a.number == 6:
            if len(mol.bonds_of(a)) == 2:
                symbols.append("C1")
            elif len(mol.bonds_of(a)) == 3:
                symbols.append("C2")
            else:
                symbols.append("C3")
        elif a.number == 14:
            symbols.append("Si")
        elif a.number == 53:
            symbols.append("I")
        else:
            raise Exception(f"Unsupported atomic number: {a.number}")
    cartesian = np.array([a.xyz for a in mol.atoms])

    # optimize a single conformer by GULP
    header = "molecule optimise gasteiger\n\n" \
        "library dreiding\n" \
        "xtol 1e-8\n" \
        "ftol 1e-8\n" \
        "gtol 1e-7\n" \
        "gmax 1e-6\n\n" \
        "species\n" \
        "H H_\n" \
        "C1 C_1\n" \
        "C2 C_R\n" \
        "C3 C_3\n" \
        "Si Si3\n" \
        "I I_\n\n"
    connect = []
    for b in mol.bonds:
        connect.append(
            f"connect {mol.index(b.atoms[0])+1} {mol.index(b.atoms[1])+1}")
    header += "\n".join(connect)
    job = gulp.Job(header=header, symbols=symbols, cartesian=cartesian)
    # with open("csp2.gin", "w") as f:
    #     f.write(job.to_gin())
    job.run()
    # with open("csp2.gout", "w") as f:
    #     f.write(job.result)
    cartesian[:, :] = job.cartesian
    for a, c in zip(mol.atoms, cartesian):
        a.xyz[:] = c[:]


def generate_lattice(minvol=1000.0, maxvol=2000.0, anisotropy=5.0):
    lattice = chem.Lattice()
    lattice.ax = 1.0
    lattice.cz = lattice.ax * random.uniform(1.0, anisotropy)
    lattice.by = random.uniform(lattice.ax, lattice.cz)
    lattice.bx = lattice.ax * random.uniform(-0.5, 0.5)
    lattice.cx = lattice.ax * random.uniform(-0.5, 0.5)
    lattice.cy = lattice.by * random.uniform(-0.5, 0.5)
    lattice.ay = 0.0
    lattice.az = 0.0
    lattice.bz = 0.0
    lattice.vectors *= math.pow(random.uniform(minvol,
                                               maxvol) / lattice.volume(), 1.0/3.0)
    return lattice


def rotation_matrix_x(angle):
    """
    returns rotation matrix which rotate about x axis by the given angle (unit: radian)
    """
    cos = math.cos(angle)
    sin = math.sin(angle)
    return np.array([[1.0,  0.0,  0.0],
                     [0.0,  cos,  sin],
                     [0.0, -sin,  cos]])


def rotation_matrix_y(angle):
    """
    returns rotation matrix which rotate about y axis by the given angle (unit: radian)
    """
    cos = math.cos(angle)
    sin = math.sin(angle)
    return np.array([[cos,  0.0, -sin],
                     [0.0,  1.0,  0.0],
                     [sin,  0.0,  cos]])


def rotation_matrix_z(angle):
    """
    returns rotation matrix which rotate about z axis by the given angle (unit: radian)
    """
    cos = math.cos(angle)
    sin = math.sin(angle)
    return np.array([[cos,  sin,  0.0],
                     [-sin,  cos,  0.0],
                     [0.0,  0.0,  1.0]])


def rotation_matrix_random():
    """
    returns rotation matrix which rotate at random
    """
    theta = math.acos(random.uniform(-1.0, 1.0))
    phi = random.uniform(0.0, 2*math.pi)
    omega = random.uniform(0.0, 2*math.pi)
    m1 = rotation_matrix_z(-theta)
    m2 = rotation_matrix_y(-phi)
    m3 = rotation_matrix_z(omega)
    m4 = rotation_matrix_y(phi)
    m5 = rotation_matrix_z(theta)
    return m5 @ m4 @ m3 @ m2 @ m1


def rotate_at_random(mol: chem.FlexMol, copy=True):
    if copy:
        mol = mol.copy()
    center = mol.center()
    mol.translate(-center)
    rot = rotation_matrix_random()
    for a in mol.atoms:
        a.xyz[:] = rot @ a.xyz
    mol.translate(center)
    return mol


def generate_crystal(lattice: chem.Lattice, mol: chem.FlexMol, n, tol=1.0, pack=True):

    def check_collision(crystal: chem.FlexMol, mol: chem.FlexMol):
        for i, j, k in itertools.product((-1, 0, 1), (-1, 0, 1), (-1, 0, 1)):
            for a in crystal.atoms:
                for b in mol.atoms:
                    if LA.norm(a.xyz + crystal.lattice.vectors.T @ (i, j, k) - b.xyz) < (a.vdw_radius + b.vdw_radius) * tol:
                        return True
            if not i == j == k == 0:
                for a in mol.atoms:
                    for b in mol.atoms:
                        if LA.norm(a.xyz + crystal.lattice.vectors.T @ (i, j, k) - b.xyz) < (a.vdw_radius + b.vdw_radius) * tol:
                            return True

        return False

    def check_collision2(crystal: chem.FlexMol):
        mols = crystal.split()
        for i, j, k in itertools.product((-1, 0, 1), (-1, 0, 1), (-1, 0, 1)):
            for p in range(len(mols)):
                for q in range(p, len(mols)):
                    if not (p == q and i == j == k == 0):
                        mol1 = mols[p]
                        mol2 = mols[q]
                        for a in mol1.atoms:
                            for b in mol2.atoms:
                                if LA.norm(a.xyz + crystal.lattice.vectors.T @ (i, j, k) - b.xyz) < (a.vdw_radius + b.vdw_radius) * tol:
                                    return True
        return False

    center_mol = mol.copy()
    center_mol.translate(-center_mol.center())

    mols = [center_mol.copy()]
    for i in range(n-1):
        rotate_at_random(center_mol)
        mols.append(center_mol.copy())

    collision = True

    for j in range(100):

        crystal = chem.FlexMol()
        crystal.lattice = lattice

        for i, new_mol in enumerate(mols):
            if i == 0:
                crystal.add_mol(new_mol)
            else:
                random_vector = lattice.vectors @ [
                    random.uniform(-1.0, 1.0) for _ in range(3)]
                for scale in np.arange(0.1, 1.05, 0.05):
                    copy_mol = new_mol.copy()
                    copy_mol.translate(scale * random_vector)
                    collision = check_collision(crystal, copy_mol)
                    if not collision:
                        print(f"mol: {i}, scale: {scale:.2f}")
                        crystal.add_mol(copy_mol)
                        break
                if collision:
                    break
            # collision = check_collision(crystal, new_mol)
            # if collision:
            #     break
            # crystal.add_mol(new_mol)

        if not collision:
            # print(f"ok {j} ", end="")
            break

    if collision:
        raise Exception("The number of collisions exceeded limit.")

    if pack:
        # try to shrink unit cell
        for axis in range(3):
            for s in range(10, 21):
                scale = s * 0.05
                new_crystal = crystal.copy()
                new_crystal.lattice.vectors[axis] *= scale
                # new_crystal.atoms = []
                # new_crystal.bonds = []
                # for mol in crystal.split():
                #     mol.translate((scale-1.0) * mol.center())
                #     new_crystal.add_mol(mol)
                if not check_collision2(new_crystal):
                    print(f"axis: {axis}, scale: {scale:.2f}")
                    # return new_crystal
                    crystal = new_crystal
                    break

    return crystal


def generate_cluster(mol: chem.FlexMol, n, tol=1.0) -> chem.FlexMol:

    def random_unit_vector():
        theta = math.acos(random.uniform(-1.0, 1.0))
        phi = random.uniform(0.0, 2*math.pi)
        s = math.sin(theta)
        return np.array((s * math.cos(phi), s * math.sin(phi), math.cos(theta)))

    def check_collision(cluster: chem.FlexMol, new_mol: chem.FlexMol):
        for a in cluster.atoms:
            for b in new_mol.atoms:
                if LA.norm(a.xyz - b.xyz) < (a.vdw_radius + b.vdw_radius) * tol:
                    return True
        return False

    def append_to_cluster(cluster: chem.FlexMol, new_mol: chem.FlexMol):
        cluster = cluster.copy()
        new_mol = new_mol.copy()
        cluster.translate(-cluster.center())
        new_mol.translate(-new_mol.center())
        # typical length of molecule
        length = statistics.mean(
            [statistics.pstdev([a.xyz[i] for a in new_mol.atoms]) for i in range(3)])
        vector = random_unit_vector()
        for _ in range(100):
            new_mol.translate(vector * length)
            if not check_collision(cluster, new_mol):
                cluster.add_mol(new_mol)
                return cluster
        return None

    center_mol = mol.copy()
    cluster = rotate_at_random(center_mol)
    for _ in range(n-1):
        new_mol = rotate_at_random(center_mol)
        cluster = append_to_cluster(cluster, new_mol)
    return cluster


def generate_crystal_from_cluster(cluster: chem.FlexMol, tol=1.0) -> chem.FlexMol:

    def check_collision(crystal: chem.FlexMol):
        mols = crystal.split()
        for i, j, k in itertools.product((-1, 0, 1), (-1, 0, 1), (-1, 0, 1)):
            for p in range(len(mols)):
                for q in range(p, len(mols)):
                    if not (p == q and i == j == k == 0):
                        mol1 = mols[p]
                        mol2 = mols[q]
                        for a in mol1.atoms:
                            for b in mol2.atoms:
                                if LA.norm(a.xyz + crystal.lattice.vectors.T @ (i, j, k) - b.xyz) < (a.vdw_radius + b.vdw_radius) * tol:
                                    return True
        return False

    # typical length of cluster
    length = statistics.mean(
        [statistics.pstdev([a.xyz[i] for a in cluster.atoms]) for i in range(3)])

    lattice = chem.Lattice()
    lattice.ax = 1.0
    lattice.by = 1.0
    lattice.cz = 1.0
    lattice.bx = lattice.ax * random.uniform(-0.5, 0.5)
    lattice.cx = lattice.ax * random.uniform(-0.5, 0.5)
    lattice.cy = lattice.by * random.uniform(-0.5, 0.5)
    lattice.ay = 0.0
    lattice.az = 0.0
    lattice.bz = 0.0
    lattice.vectors *= length * 3
    cluster.lattice = lattice

    for _ in range(20):
        print(".", end="", flush=True)
        if not check_collision(cluster):
            break
        lattice.vectors *= 1.1

    ok = np.full(3, True, dtype=bool)
    for _ in range(100):
        for axis in range(3):
            if ok[axis]:
                print("*", end="", flush=True)
                lattice.vectors[axis] *= 0.9
                if check_collision(cluster):
                    lattice.vectors[axis] /= 0.9
                    ok[axis] = False
        if not np.any(ok):
            break
    return cluster


def lem(crystal: chem.FlexMol):
    cartesian = np.array([a.xyz for a in crystal.atoms])
    numbers = np.array([a.number for a in crystal.atoms], dtype=int)
    symbols = [atpro.SYMBOLS[n] for n in numbers]

    charges = np.zeros(len(numbers))
    for i, n in enumerate(numbers):
        if n == 1:
            charges[i] = 0.109
        else:
            charges[i] = -0.109

    header = "molecule c6 optimise conp\n\n" \
        f"output cif {crystal.name}_final\n" \
        "library dreiding\n\n" \
        "species\n" \
        "C C_R\n" \
        "H H_\n\n"
    connect = []
    for b in crystal.bonds:
        connect.append(
            f"connect {crystal.index(b.atoms[0])+1} {crystal.index(b.atoms[1])+1}")
    header += "\n".join(connect)
    job = gulp.Job(header=header, vectors=crystal.lattice.vectors,
                   symbols=symbols, cartesian=cartesian, charges=charges)
    with open(f"{crystal.name}.gin", "w") as f:
        f.write(job.to_gin())
    job.run()
    with open(f"{crystal.name}.gout", "w") as f:
        f.write(job.result)
    return job.error_termination


if __name__ == "__main__":

    # create a single conformer
    # smiles = "CC(C)(C)[Si](C#Cc4c2cc1ccccc1cc2c(C#C[Si](C(C)(C)C)(C(C)(C)C)C(C)(C)C)c5cc3cc(I)c(I)cc3cc45)(C(C)(C)C)C(C)(C)C"

    mol = gen3d("c1ccccc1")
    optimize_conformer(mol)
    mol.to("csp2.mol")

    def task(name):
        time.sleep(random.uniform(1.0, 10.0))
        print(f"{name} start at {datetime.datetime.now()}")
        cluster = generate_cluster(mol, 2)
        crystal = generate_crystal_from_cluster(cluster)
        crystal.name = name
        crystal.to(f"{name}.cif")
        lem(crystal)
        print(f"{name} end at {datetime.datetime.now()}")

    names = [f"benzene_{i:06}" for i in range(10000)]

    with ProcessPoolExecutor(max_workers=int(sys.argv[1])) as executor:
        executor.map(task, names)
