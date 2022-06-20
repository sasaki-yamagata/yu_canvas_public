"""Physical constants"""
import math
from matsuilab import unit

pi = math.pi
"""Pi"""

h = 6.62607015e-34 * unit.J * unit.s
"""Planck constant"""

hbar = h / (2.0 * pi)
"""Planck constant over 2*pi"""

k_b = 1.380649e-23 * unit.J / unit.K
"""Boltzmann constant"""

n_a = unit.mol
"""Avogadro constant"""

e = 1.602176634e-19 * unit.C
"""Elementary charge"""

c = 2.99792458e8 * unit.m / unit.s
"""Light speed"""

epsilon_0 = 8.8541878128e-12 * unit.F / unit.m
"""Permittivity of vacuum"""

bohr_radius = 5.29177210903e-11 * unit.m