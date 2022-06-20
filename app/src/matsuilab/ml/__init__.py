"""
機械学習モジュール

主な機能
    * gcdファイルの読み書き
    * 一次スクリーニング
    * Gaussianジョブファイルの作成、結果の読み取り
    * Gaussian Cubeファイルの読込みとマスク処理
    * 部分木構造に関する処理

Examples:
    2つのgcdファイルの和集合、積集合、差集合を取り出す。

    >>> from matsuilab import ml
    >>> a = set(ml.read_gcd("a.gcd")) # 読み込んで、set型に変換
    >>> b = set(ml.read_gcd("b.gcd"))
    >>> union = a | b # 和集合（aまたはbに含まれる）
    >>> print(len(union))
    >>> ml.write_gcd(union, "union.gcd")
    >>> difference = a - b # 差集合（aに含まれるが、bに含まれない）
    >>> print(len(difference))
    >>> ml.write_gcd(difference, "difference.gcd")
    >>> intersection = a & b # 積集合（aにもbにも含まれる）
    >>> print(len(intersection))
    >>> ml.write_gcd(intersection, "intersection.gcd")

    gcdファイルのデータから、単一成分のデータだけをスクリーニングする。

    >>> from matsuilab import ml
    >>> old_list = ml.read_gcd("old_list.gcd")
    >>> new_list = []
    >>> for refcode in old_list:
    >>>     if ml.number_of_chemical_species(refcode) == 1:
    >>>         new_list.append(refcode)
    >>> print(len(old_list))
    >>> print(len(new_list))
    >>> ml.write_gcd(new_list, "new_list.gcd")

    gcdファイルに含まれる全ての分子のGaussianジョブファイルを作成する。

    >>> from matsuilab import ml
    >>> refcodes = ml.read_gcd("a.gcd")
    >>> count = 0
    >>> for ref in refcodes:
    >>>     ml.refcode_to_gjf(ref, gjf_file=ref+".gjf", nproc=36, mem="160GB")
    >>>     count += 1
    >>> print(count)

    gcdファイルに含まれる全ての分子の部分木構造記述子の出現回数をrefcode(行)と記述子(列)の表に集計する。

    >>> refcodes = ml.read_gcd("example.gcd") # refcodeのリスト
    >>> descriptors = list(ml.tree_descriptor_counts(refcodes, 2).keys()) # 記述子のリスト
    >>> descriptors.sort()
    >>> dataframe = ml.tree_descriptor_data_frame(refcodes, 2, descriptors)
    >>> dataframe.to_pickle("example.pkl")
    >>> print(dataframe[0:2])
"""

import math
import os
import random
import subprocess
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from matsuilab import atpro, const, unit
from scipy import sparse


def read_gcd(gcd_file):
    """
    gcdファイルからrefcodeの一覧を読み込む。
    和集合、差集合、積集合を求めるときは、set型に変換すると便利。

    Args:
        gcd_file (str): gcdファイル名

    Returns:
        list[str]: refcodeの一覧
    """
    refcodes = []
    with open(gcd_file, "r") as f:
        for line in f:
            refcodes.append(line.strip())
    return refcodes


def write_gcd(refcodes, gcd_file):
    """
    refcodeの一覧をgcdファイルに書き出す。

    Args:
        refcodes (list[str]): refcodeの一覧
        gcd_file (str): gcdファイル名
    """
    with open(gcd_file, "w") as f:
        for ref in refcodes:
            f.write(ref+"\n")


def number_of_chemical_species(refcode):
    """
    指定したrefcodeのデータに含まれる化学種の数を返す。

    Args:
        refcode (str): refcode

    Returns:
        int: 化学種の数
    """
    from ccdc import io

    formulas = set()
    for mol in io.MoleculeReader("CSD").molecule(refcode).components:
        formulas.add(mol.formula)  # set型なので、重複しないものだけ加わる。
    return len(formulas)


def refcode_to_gjf(refcode, gjf_file, header=None, nproc=None, mem=None, method="b3lyp", basis="6-31g*", title="No Title", encoding="utf-8"):
    """
    指定したrefcodeのデータから単一分子のgjfファイルを作成する。

    Args:
        refcode (str): refcode (例: "AABHTZ")
        gjf_file (str): 保存先のファイル名 (例: "AABHTZ.gjf")
        header (str): 計算条件を表す%または#から始まる全行。Noneの場合はデフォルトの計算条件を使用する。
        nproc (int): 使用するプロセッサのコア数。header=None の場合のみ有効。
        mem (str): メモリ容量 (例: "160GB"). header=None の場合のみ有効。
        method (str): 計算手法。header=None の場合のみ有効。
        basis (str): 基底関数系。header=None の場合のみ有効。
        title (str): 任意のタイトル (最大5行)
        encoding (str): 文字コード
    """
    from ccdc import io

    # 原子のリストを取得
    mol = io.MoleculeReader("CSD").molecule(refcode)
    mol.assign_bond_types(which='unknown')
    mol.add_hydrogens(mode='missing')
    atoms = mol.atoms

    # 全電子数を数えて、ラジカルかどうかを判定
    ele_num = sum([a.atomic_number for a in atoms])

    # 書き込むファイルを開く。
    with open(file=gjf_file, mode="w", encoding=encoding) as f:

        # header=None の場合は、デフォルトの計算条件を使用
        if header is None:
            header = []
            if nproc is not None:
                header.append("%nprocshared=" + str(nproc))
            if mem is not None:
                header.append("%mem=" + str(mem))
            header.append("# " + method + "/" + basis)
            header.append("\n")
            header = "\n".join(header)

        # ヘッダを出力
        f.write(header)

        # タイトルを出力
        f.write(title + "\n\n")

        # 電荷とスピン多重度を出力
        if ele_num % 2 == 0:
            # singlet
            f.write("0 1\n")
        else:
            # radical, doublet
            f.write("0 2\n")

        # 原子リストを出力
        for a in atoms:
            f.write("{0}\t{1:.6f}\t{2:.6f}\t{3:.6f}\n".format(
                a.atomic_symbol, *a.coordinates))
        f.write("\n")


def read_homo_lumo_from_log(log_file):
    """
    GaussianのlogファイルからHOMO, LUMOエネルギー(eV)を抽出する。

    Args:
        log_file (str): logファイル名

    Returns:
        tuple[float]: HOMO, LUMOエネルギー (eV)
    """
    homo = None
    lumo = None
    with open(log_file, "r") as f:
        flag = False  # 空軌道の１行目だけを読むために使用
        for line in f:
            # 被占軌道の最後の固有エネルギーを取得
            if line.startswith(" Alpha  occ. eigenvalues -- "):
                homo = line[28:].split()[-1]
                flag = True
            # 空軌道の最初の固有エネルギーを取得
            if flag and line.startswith(" Alpha virt. eigenvalues -- "):
                lumo = line[28:].split()[0]
                flag = False
    if homo is not None:
        homo = float(homo) * 27.2114
    if lumo is not None:
        lumo = float(lumo) * 27.2114
    return homo, lumo


def tree_descriptor(atom, depth):
    """
    原子の部分木構造記述子を返す。

    Args:
        atom (ccdc.molecule.Atom): 対象原子
        depth (int): 深さレベル (0: 元素記号のみ、1: 隣接原子まで、2: 2つ隣の原子まで、…)

    Returns:
        str: 部分木構造記述子
    """

    # depth = 0 なら元素記号をそのまま返す
    if depth == 0:
        return atom.atomic_symbol

    # depth > 0 なら、depth-1 の表現を組み合わせて作る。
    else:
        sub = [tree_descriptor(a, depth-1) for a in atom.neighbours]
        sub.sort()

        des = atom.atomic_symbol + "-"
        for s in sub:
            if depth == 1:
                des = des + s
            else:
                des = des + "(" + s + ")"

        return des


def tree_descriptor_counts(refcodes, depth):
    """
    全refcode内の部分木構造記述子の一覧とそれぞれの出現回数をdict型で返す。

    Args:
        refcodes (list[str]): refcodeの一覧
        depth (int): 深さレベル (0: 元素記号のみ、1: 隣接原子まで、2: 2つ隣の原子まで、…)

    Returns:
        dict[str, int]: 部分木構造記述子とその出現回数
    """
    from ccdc import io

    reader = io.MoleculeReader("CSD")
    counts = dict()
    for ref in refcodes:
        mol = reader.molecule(ref).components[0]
        mol.assign_bond_types(which='unknown')
        mol.add_hydrogens(mode='missing')
        for atom in mol.atoms:
            des = tree_descriptor(atom, depth)
            if des in counts.keys():
                counts[des] += 1
            else:
                counts[des] = 1
    return counts


def tree_descriptor_data_frame(refcodes, depth, descriptors=None):
    """
    全refcode内の部分木構造記述子の出現回数をrefcode(行)と記述子(列)の表に集計する。

    Args:
        refcodes (list[str]): refcodeの一覧
        depth (int): 深さレベル (0: 元素記号のみ、1: 隣接原子まで、2: 2つ隣の原子まで、…)
        descriptors (list[str]): 部分木構造記述子の一覧。Noneの場合は自動で生成する。

    Returns:
        pandas.DataFrame: refcode(行)と記述子(列)毎の記述子の出現回数。
    """
    if descriptors is None:
        descriptors = list(tree_descriptor_counts(refcodes, depth).keys())
        descriptors.sort()

    matrix = sparse.lil_matrix((len(refcodes), len(descriptors)), dtype=int)
    for i, ref in enumerate(refcodes):
        counts = tree_descriptor_counts([ref], depth)
        for k, c in counts.items():
            try:
                matrix[i, descriptors.index(k)] = c
            except ValueError:
                pass
    return pd.DataFrame.sparse.from_spmatrix(matrix.tocsr(), index=refcodes, columns=descriptors)


def create_gjf(infile:str, outfile:str=None, header:str="#p opt b3lyp/6-31g(d)", title:str="created by matsuilab.ml", charge:int=0, spin:int=1, footer:str="", randomize:float=None):
    """
    Create a gaussian job file (\*.gjf) from mol or log file

    Args:
        infile: input file (\*.mol or \*.log)
        outfile: output gjf file
        header: header in gjf file content
        title: title in gif file content
        charge: charge
        spin: spin multiplicity
        footer: footer in gjf file content
        randomize: shift atom positions at random
    """
    if outfile is None:
        outfile = os.path.splitext(infile)[0] + ".gjf"

    with open(outfile, "w") as gjf:
        gjf.write(f"{header}\n\n{title}\n\n{charge} {spin}\n")
        for a in get_geometry(infile):
            if randomize is not None:
                x = a[1]+random.uniform(-randomize, randomize)
                y = a[2]+random.uniform(-randomize, randomize)
                z = a[3]+random.uniform(-randomize, randomize)
                gjf.write(f" {a[0]} {x: 9.6f} {y: 9.6f} {z: 9.6f}\n")
            else:
                gjf.write(f" {a[0]} {a[1]: 9.6f} {a[2]: 9.6f} {a[3]: 9.6f}\n")
        gjf.write(footer)
        gjf.write("\n\n")


def get_geometry(file:str, asarray:bool=False) -> List[Tuple[str, float, float, float]]:
    """Get list of atom symbols and atom coordinates from mol or log file
    
    Args:
        file: mol file or log file
        asarray: If True, it returns tuple of atomic number array and coordinate array
    """
    if asarray:
        geom = get_geometry(file, asarray=False)
        numbers = np.zeros(len(geom), dtype=int)
        coordinates = np.zeros((len(geom), 3))
        for i in range(len(geom)):
            numbers[i] = atpro.SYMBOLS.index(geom[i][0])
            coordinates[i, :] = geom[i][1:]
        return numbers, coordinates
    else:
        if file.endswith(".mol"):
            return _get_geometry_from_mol(file)
        elif file.endswith(".log"):
            return _get_geometry_from_log(file)
        else:
            raise NotImplementedError()


def _get_geometry_from_mol(file):
    geom = []
    with open(file, "r") as f:
        f.readline()
        f.readline()
        f.readline()
        n_atoms = int(f.readline().strip().split()[0])
        for _ in range(n_atoms):
            tokens = f.readline().strip().split()
            geom.append((tokens[3], float(tokens[0]),
                         float(tokens[1]), float(tokens[2])))
    return geom


def _get_geometry_from_log(file):

    with open(file, "r") as f:
        found = False
        buffer = []
        for line in f:
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
        # windows
        buffer = join_buffer.split("|")
    else:
        # linux
        buffer = join_buffer.split("\\")

    blanks = [i for i, b in enumerate(buffer) if b == ""]

    buffer = buffer[blanks[2]+2:blanks[3]]

    geom = []
    for b in buffer:
        a = b.split(",")
        if len(a) == 4:
            geom.append((a[0], float(a[1]), float(a[2]), float(a[3])))
        elif len(a) == 5:
            geom.append((a[0], float(a[2]), float(a[3]), float(a[4])))
        else:
            raise NotImplementedError(a)

    return geom


class GaussianLogFile():
    """An object to read a variety of information from a Gaussian log file (.log)"""

    def __init__(self, path:str):
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

    def final_energy(self, unit:str="eV") -> float:
        """Returns final energy"""
        energy = None
        for l in self.lines:
            if l.startswith(" SCF Done:"):
                energy = float(l.split()[4])
        if unit == "eV":
            energy *= 27.2114
        return energy

    def homo_lumo_energies(self, unit:str="eV") -> Tuple[float, float]:
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
            return join_buffer.split("|") # windows
        else:
            return join_buffer.split("\\") # linux

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


def calc_multi_redox(file:str, mem:str=None, parallel:int=1, opt:str="b3lyp/6-31g(d,p)", sp1:str="b3lyp/3-21g", sp2:str="b3lyp/6-31+g(d,p)", limit:int=7, minpot:float=1.5, li0:float=-7.49120172815*27.2114, gau:str="g16"):
    """
    Calculate multiple redox potentials with adding Li ions
    
    Args:
        file: input file (.mol)
        mem: memory size to be used (ex. "1GB")
        parallel: the number of processors shared
        opt: method and basis set for structural optimization
        sp1: rough method and basis set for single point calculation
        sp2: fine method and basis set for single point calculation
        limit: the maximum number of lithium ions
        minpot: minimum potential required
        li0: energy of a neutral litium atom in eV
    """

    def delete_file(file):
        if os.path.exists(file):
            os.remove(file)

    def rename_file(old_file, new_file):
        if os.path.exists(old_file):
            delete_file(new_file)
            os.rename(old_file, new_file)

    def print_log(log):
        print(log.path)
        print(f" Final energy: {log.final_energy()}")
        print(f" Job cpu time: {log.job_cpu_time()}")
        print(f" Elapsed time: {log.elapsed_time()}", flush=True)

    energies = []

    basename, ext = os.path.splitext(file)

    if mem is None:
        mem = ""
    else:
        mem = f"%mem={mem}\n"
    header_opt = f"{mem}%nprocshared={parallel}\n#p {opt} opt=tight"
    header_sp  = f"{mem}%nprocshared={parallel}\n#p {sp1}"
    footer_sp  = f"{mem}%nprocshared={parallel}\n#p {sp2} geom=allcheck guess=checkpoint scf=tight\n"

    adjacency = adjacency_matrix_from_coordinates(*get_geometry(file, asarray=True))

    # 半経験的手法で構造最適化
    # xxx.mol > xxx_0_opt.gjf, xxx_0_opt.log
    create_gjf(file, f"{basename}_0_opt.gjf", header=header_opt, randomize=1e-3)
    subprocess.run([gau, f"{basename}_0_opt.gjf"])
    rename_file(f"{basename}_0_opt.out", f"{basename}_0_opt.log")

    log = GaussianLogFile(f"{basename}_0_opt.log")
    if not log.normal_termination():
        return
    else:
        print_log(log)

    # DFT, 中性状態でエネルギー計算
    # xxx_0_opt.log > xxx_0_sp.gjf, xxx_0_sp.log
    create_gjf(f"{basename}_0_opt.log", f"{basename}_0_sp.gjf",
               header=f"%chk={basename}_0_sp.chk\n{header_sp}",
               footer=f"\n--Link1--\n%chk={basename}_0_sp.chk\n{footer_sp}")
    subprocess.run([gau, f"{basename}_0_sp.gjf"])
    rename_file(f"{basename}_0_sp.out", f"{basename}_0_sp.log")

    delete_file(f"{basename}_0_sp.chk")

    log = GaussianLogFile(f"{basename}_0_sp.log")
    if not log.normal_termination():
        return
    else:
        print_log(log)

    energies.append(GaussianLogFile(f"{basename}_0_sp.log").final_energy())

    for i in range(1, limit+1):

        # DFT, アニオン状態で静電ポテンシャルを計算
        # xxx_(i-1)_sp.log > xxx_i_esp.gjf, xxx_i_esp.log, xxx_i_esp.chk
        create_gjf(f"{basename}_{i-1}_sp.log", f"{basename}_{i}_esp.gjf",
                   header=f"%chk={basename}_{i}_esp.chk\n{header_sp}", charge=-1, spin=i % 2+1,
                   footer=f"\n--Link1--\n%chk={basename}_{i}_esp.chk\n{footer_sp}")
        subprocess.run([gau, f"{basename}_{i}_esp.gjf"])
        rename_file(f"{basename}_{i}_esp.out", f"{basename}_{i}_esp.log")

        log = GaussianLogFile(f"{basename}_{i}_esp.log")
        if not log.normal_termination():
            return
        else:
            print_log(log)

        # リチウムを追加
        # xxx_i_esp.chk > xxx_i_esp.fchk, xxx_i_esp.cube, xxx_i_den.cube, Li-position
        subprocess.run(
            ["formchk", f"{basename}_{i}_esp.chk", f"{basename}_{i}_esp.fchk"])
        subprocess.run(["cubegen", str(parallel), "Potential=SCF",
                        f"{basename}_{i}_esp.fchk", f"{basename}_{i}_esp.cube", "-2", "h"])
        subprocess.run(["cubegen", str(parallel), "Density=SCF",
                        f"{basename}_{i}_esp.fchk", f"{basename}_{i}_den.cube", "-2", "h"])
        if os.name == "nt":
            command = os.path.join(os.path.dirname(sys.modules[__name__].__file__), "fortran", "lithiumization.exe")
        else:
            command = os.path.join(os.path.dirname(sys.modules[__name__].__file__), "fortran", "lithiumization")
        result = subprocess.run([command, f"{basename}_{i}_den.cube", f"{basename}_{i}_esp.cube"],
                                stdout=subprocess.PIPE, encoding="utf-8")
        lines = result.stdout.split("\n")
        atoms = []
        for l in lines[lines.index("For Pasting in GJF File:")+1:]:
            s = l.split()
            if len(s) == 4:
                atoms.append((s[0],
                              float(s[1]) + random.uniform(-1e-3, 1e-3),
                              float(s[2]) + random.uniform(-1e-3, 1e-3),
                              float(s[3]) + random.uniform(-1e-3, 1e-3)))

        delete_file(f"{basename}_{i}_esp.chk")
        delete_file(f"{basename}_{i}_esp.fchk")
        delete_file(f"{basename}_{i}_esp.cube")
        delete_file(f"{basename}_{i}_den.cube")

        # 半経験的手法で構造最適化
        # xxx_i_esp.log > xxx_i_opt.gjf, xxx_i_opt.log
        with open(f"{basename}_{i}_opt.gjf", "w") as f:
            f.write(
                f"{header_opt}\n\ncreated by matsuilab.ml\n\n0 {i%2+1}\n")
            for a in atoms:
                f.write(f" {a[0]} {a[1]: 9.6f} {a[2]: 9.6f} {a[3]: 9.6f}\n")
            f.write("\n")
        subprocess.run([gau, f"{basename}_{i}_opt.gjf"])
        rename_file(f"{basename}_{i}_opt.out", f"{basename}_{i}_opt.log")

        log = GaussianLogFile(f"{basename}_{i}_opt.log")
        if not log.normal_termination():
            return
        else:
            print_log(log)

        # DFT, 中性状態でエネルギー計算
        # xxx_i_opt.log > xxx_i_sp.gjf, xxx_i_sp.log
        create_gjf(f"{basename}_{i}_opt.log", f"{basename}_{i}_sp.gjf",
                   header=f"%chk={basename}_{i}_sp.chk\n{header_sp}", spin=i % 2+1,
                   footer=f"\n--Link1--\n%chk={basename}_{i}_sp.chk\n{footer_sp}")
        subprocess.run([gau, f"{basename}_{i}_sp.gjf"])
        rename_file(f"{basename}_{i}_sp.out", f"{basename}_{i}_sp.log")
        # subprocess.run(
        #     ["formchk", f"{basename}_{i}_sp.chk", f"{basename}_{i}_sp.fchk"])

        delete_file(f"{basename}_{i}_sp.chk")

        log = GaussianLogFile(f"{basename}_{i}_sp.log")
        if not log.normal_termination():
            return
        else:
            print_log(log)

        energies.append(GaussianLogFile(f"{basename}_{i}_sp.log").final_energy())

        # calculate redox potential (vs Li/Li+)
        redox = energies[i-1] + li0 - energies[i]
        print(f"{i}-th redox potential of {basename} is {redox} V", flush=True)

        new_adjacency = adjacency_matrix_from_coordinates(*get_geometry(f"{basename}_{i}_sp.log", asarray=True))
        if not np.all(adjacency == new_adjacency[:adjacency.shape[0], :adjacency.shape[1]]):
            print(f"Connectivity has changed. ({basename}, {i})", flush=True)
            return
        elif redox < minpot:
            print(f"Termination due to low redox porential < {minpot} V. ({basename}, {i})", flush=True)
            return


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

    def __init__(self, path:str):
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
            x1 = min(math.ceil((coord[0] + r - self.origin[0]) / dx) + 1, self.npts[0])
            y0 = max(math.floor((coord[1] - r - self.origin[1]) / dy), 0)
            y1 = min(math.ceil((coord[1] + r - self.origin[1]) / dy) + 1, self.npts[1])
            z0 = max(math.floor((coord[2] - r - self.origin[2]) / dz), 0)
            z1 = min(math.ceil((coord[2] + r - self.origin[2]) / dz) + 1, self.npts[2])
            mask[x0:x1, y0:y1, z0:z1] = np.logical_or(mask[x0:x1, y0:y1, z0:z1], np.linalg.norm(mesh_coordinates[x0:x1, y0:y1, z0:z1, :] - coord[:], axis=3) <= r)

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


def is_bonded(num1:int, num2:int, distance:float, threshold:float=1.2) -> bool:
    """
    Checks whether the two atoms are bonded, judging from the distance
    
    Args:
        num1: atomic number of first atom
        num2: atomic number of second atom
        distance: distance between the two atoms
        threshold: judged as bonded if distance is less than (the sum of single covalent radii) * threshold
    """
    return distance < (atpro.SINGLE_COVALENT_RADII[num1] + atpro.SINGLE_COVALENT_RADII[num2]) * threshold


def adjacency_matrix_from_coordinates(numbers:np.ndarray, coordinates:np.ndarray) -> np.ndarray:
    """Create adjacency matrix from atomic numbers and cartesian coordinates
    
    Args:
        numbers: 1D array of atomic numbers of all atoms
        coordinates: 2D array of cartesian coordinates of all atoms
    """
    n = len(numbers)
    adjacency = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i):
            adjacency[i, j] = is_bonded(numbers[i], numbers[j], np.linalg.norm(coordinates[i, :] - coordinates[j, :]))
            adjacency[j, i] = adjacency[i, j]
    return adjacency
