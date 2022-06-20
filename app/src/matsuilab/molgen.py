"""Molecular structure generator using Monte Carlo tree search (MCTS).
The main class *MolGen* generates new nodes.
A *Node* object has attributes including mol (a generated molecular structure as FlexMol), score, depth.

Example:
    >>> molgen = MolGen()
    >>> for i, node in enumerate(MolGen().generate_multiple()):
    >>>     node.mol.to(f"{i}.png")

Notes:
    Users can tune the behavior by the arguments of *generate_multiple* function and the following class variables.

    * AddAtomNode.elements and AddAtomNode.elements_weights
    * AddAtomNode.bond_types and AddAtomNode.bond_weights
    * AddBondNode.bond_types and AddBondNode.bond_weights
    * MolGen.ucb_const
    * MolGen.average_depth
    * MolGen.node_weights

    Advanced users can also customize the followings.

    * define subclasses of Node, and add to MolGen.node_classes
    * override MolGen._select(self) -> Node
    * override MolGen._expand(self, node: Node) -> Node
    * override MolGen._evaluate(self, mol) -> float
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple, Generator

import numpy as np
from openbabel import openbabel as ob

from matsuilab import atpro
from matsuilab.chem import FlexMol


class Node():
    """node in Monte Carlo tree search

    Attributes:
        parent: parent node
        children: list of child nodes
        mol: molecular structure
        score: score
        score_sum: sum of scores of this and child nodes
        count: sum of trials for this and child nodes
        depth: depth of this node in the tree structure. Root node has depth=0.
    """

    def __init__(self, parent: Node):
        self.parent: Node = parent
        self.children: List[Node] = []
        self.mol: FlexMol = None
        self.score: float = 0.0
        self.score_sum: float = 0.0
        self.count: int = 0
        self.depth: int = 0
        if parent is not None:
            self.depth = parent.depth + 1

    def __eq__(self, other):
        """check whether *self* is equivalent to *other*"""
        raise NotImplementedError(
            "subclass of Node must implement __eq__(self, other)")

    @classmethod
    def _create(cls, node: Node):
        """create new child node at *node*"""
        raise NotImplementedError(
            "subclass of Node must implement a class method _create(cls, node: Node)")

    def _apply(self, mol: FlexMol):
        """modify molecular structure"""
        raise NotImplementedError(
            "subclass of Node must implement _apply(self, mol: FlexMol)")


class RootNode(Node):

    def __init__(self, mol):
        super().__init__(None)
        self.mol = FlexMol(mol)
        self.mol.add_hydrogens()


class AddAtomNode(Node):
    """node which adds an atom of *element* at *position* with *bond_type*"""

    elements = (5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 52, 53)
    """class variable. list of atomic numbers to be chosen at random"""

    element_weights = (119, 22049, 1799, 3015, 222, 105,
                       89, 257, 166, 14, 109, 4, 22)
    """class variable. relative probabilities of atomic numbers"""

    bond_types = (1, 2, 3)
    """class variable. bond types to be chosen at random"""

    bond_weights = (10, 5, 1)
    """class variable. relative probabilities of bond types"""

    @classmethod
    def _choose_element(cls):
        return random.choices(cls.elements, cls.element_weights)[0]

    @classmethod
    def _choose_position(cls, mol: FlexMol):
        for atom in random.sample(mol.atoms, len(mol.atoms)):
            if mol.num_hydrogens(atom) > 0:
                return mol.index(atom)
        return -1

    @classmethod
    def _choose_bond_type(cls, mol: FlexMol, position: int, element: int):
        weights = [10, 8, 1]
        max_bond = min(mol.num_hydrogens(position),
                       4 - abs(4 - atpro.VALENCE_ELECTRONS[element]))
        max_bond = max(1, max_bond)
        max_bond = min(3, max_bond)
        return random.choices(cls.bond_types[:max_bond], cls.bond_weights[:max_bond])[0]

    def __init__(self, parent: Node, position: int, element: int, bond_type: int = 1):
        self.position: int = position
        self.element: int = element
        self.bond_type: int = bond_type
        super().__init__(parent)

    def __eq__(self, other):
        if isinstance(other, AddAtomNode):
            return self.position == other.position and self.element == other.element and self.bond_type == other.bond_type
        else:
            return False

    @classmethod
    def _create(cls, node: Node):
        element = cls._choose_element()
        position = cls._choose_position(node.mol)
        if position >= 0:
            bond_type = cls._choose_bond_type(node.mol, position, element)
            new_node = AddAtomNode(node, position, element, bond_type)
            if new_node not in node.children:
                return new_node
        return None

    def _apply(self, mol: FlexMol):
        mol.attach_atom(self.position, self.element, self.bond_type)
        return mol


class AddBondNode(Node):
    """node which adds a bond of *bond_type* between the atoms of *positions*"""

    bond_types = (1, 2, 3)
    """class variable. bond types to be chosen at random"""

    bond_weights = (10, 5, 1)
    """class variable. relative probabilities of bond types"""

    @classmethod
    def _choose_positions(cls, mol: FlexMol):

        candidates = []
        for a in mol.atoms:
            if a.number != 1 and mol.num_hydrogens(a) > 0:
                candidates.append(a)

        random.shuffle(candidates)

        for i in range(len(candidates)):
            for j in range(i):
                if candidates[j] not in mol.neighbors_of(candidates[i]):
                    return mol.index(candidates[i]), mol.index(candidates[j])

        return None

    @classmethod
    def _choose_bond_type(cls, mol: FlexMol, positions: Tuple[int, int]):
        max_bond = min(*map(mol.num_hydrogens, positions), 3)
        return random.choices(cls.bond_types[:max_bond], cls.bond_weights[:max_bond])[0]

    def __init__(self, parent: Node, positions: Tuple[int, int], bond_type: int = 1):
        self.positions: int = positions
        self.bond_type: int = bond_type
        super().__init__(parent)

    def __eq__(self, other):
        if isinstance(other, AddBondNode):
            return set(self.positions) == set(other.positions) and self.bond_type == other.bond_type
        else:
            return False

    @classmethod
    def _create(cls, node: Node):
        positions = cls._choose_positions(node.mol)
        if positions is not None:
            bond_type = cls._choose_bond_type(node.mol, positions)
            new_node = AddBondNode(node, positions, bond_type)
            if new_node not in node.children:
                return new_node
        return None

    def _apply(self, mol: FlexMol):
        atoms = [mol.atoms[p] for p in self.positions]
        for a in atoms:
            mol.detach_hydrogens(a, self.bond_type)
        mol.add_bond(*atoms, self.bond_type)
        return mol


class MolGen():
    """
    molecular structure generator.
    *mol* is the initial molecular structure at root node.
    If *mol* is None, methane is used.
    """

    ucb_const = math.sqrt(2)
    """constant parameter in UCB definition"""

    average_depth = 20.0
    """average depth of new nodes to create"""

    node_classes = (AddAtomNode, AddBondNode)
    """list of node classes"""

    node_weights = (0.9, 0.1)
    """probabilities of choosing each node class"""

    def __init__(self, mol=None):

        if mol is None:
            mol = FlexMol("C")

        self.root: RootNode = RootNode(mol)
        """root node of tree"""

    def generate(self) -> Node:
        """generates a single molecular structure."""
        selected = self._select()
        new_node: Node = self._expand(selected)
        if new_node is None:
            self._backpropagate(selected, 0.0)
        else:
            if new_node not in selected.children:
                selected.children.append(new_node)
            if new_node.mol is None:
                new_node.mol = new_node._apply(new_node.parent.mol.copy())
            new_node.score = self._evaluate(new_node.mol)
            self._backpropagate(new_node, new_node.score)
        return new_node

    def generate_multiple(self, num: int = 10, min_score: float = 1.0, max_cycle: int = 10000, max_fail: int = 1000, logging: bool = True) -> Generator[Node]:
        """
        generates multiple molecular structures. It actually returns a generator object.
        To obtain the results as a list object, use as follows.
        
        >>> results = [r for r in self.generate_multiple()]

        Args:
            num: the number of final outputs
            min_score: minimum score required
            max_cycle: maximum number of cycles
            max_fail: maximum number of failure accepted
            logging: whether log is printed on screen
        """
        failed = 0
        counts = 0
        for i in range(max_cycle):
            new_node = self.generate()
            if new_node is None:
                if logging:
                    print(f"cycle: {i:6d}, failed")
                failed += 1
                if failed >= max_fail:
                    break
            else:
                if new_node.score > min_score:
                    if logging:
                        print(f"cycle: {i:6d}, depth: {new_node.depth:3d}, score: {new_node.score:6.3f}, Good")
                    counts += 1
                    yield new_node
                else:
                    if logging:
                        print(f"cycle: {i:6d}, depth: {new_node.depth:3d}, score: {new_node.score:6.3f}")
                if counts >= num:
                    break

    def _select(self) -> Node:
        """
        select an existing node to grow.
        This function may be overwritten for customized search.
        """

        selected = self.root

        while len(selected.children) > 0 and random.random() >= 1.0 / self.average_depth:
            ucb = np.zeros(len(selected.children))
            for i, child in enumerate(selected.children):
                ucb[i] = child.score_sum / child.count + \
                    self.ucb_const * math.log(selected.count) / child.count
            selected = selected.children[np.argmax(ucb)]

        return selected

    def _expand(self, node: Node) -> Node:
        """
        creates and returns a new child node just below the *node*.
        It returns None if failed.
        This function may be overwritten for customized search.
        """

        for _ in range(5):
            node_class = random.choices(
                self.node_classes, weights=self.node_weights)[0]
            new_node = node_class._create(node)
            if new_node is not None and new_node not in node.children:
                return new_node

        return None

    def _evaluate(self, mol) -> float:
        """
        calculate the score of *mol*.
        This function may be overwritten for customized search.
        """
        score = 0.0
        natoms = 0
        for a in mol.atoms:
            if sum([b.bond_type == 2 for b in mol.bonds_of(a)]) == 1:
                score += 1.0
            if a.number != 1:
                natoms += 1
        obmol: ob.OBMol = mol.to("ob")
        obmol.SetAromaticPerceived()
        for atom in ob.OBMolAtomIter(obmol):
            if atom.IsAromatic():
                if atom.IsInRingSize(5):
                    score += 1.0
                if atom.IsInRingSize(6):
                    score += 3.0
        return score / natoms

    def _backpropagate(self, node: Node, score: float):
        node.score_sum += score
        node.count += 1
        if node.parent is not None:
            self._backpropagate(node.parent, score)


if __name__ == "__main__":

    for i in range(10):
        results = MolGen().generate_multiple(10)
        for j, r in enumerate(results):
            r.mol.to(f"{i}-{j}.png")
