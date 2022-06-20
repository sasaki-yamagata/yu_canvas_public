"""
Unit conversions

Example:
    import

    >>> from matsuilab import unit

    convert Angstrom to meter

    >>> value_in_A = 1.0
    >>> value_in_m = value_in_A * unit.Angstrom / unit.m
"""
import math

# time
s = 1.0
"""Second"""

# length
m = 1.0
"""Meter"""

Angstrom = 1e-10 * m
"""Angstrom"""

# mass
kg = 1.0
"""Kilo-gram"""

g = 1e-3 * kg
"""Gram"""

Da = 1.66053906660e-27 * kg
"""Dalton"""

# current
A = 1.0
"""Ampere"""

# temperature
K = 1.0
"""Kelvin"""

# energy
J = 1.0
"""Joule"""

kJ = 1e3 * J
"""Kilo-joule"""

cal = 4.184
"""Calorie"""

kcal = 1e3 * cal
"""Kilo-calorie"""

eV = 1.6022e-19
"""Electron volt"""

# charge
C = A * s
"""Coulomb"""

# amount of substance
mol = 6.02214076e23
"""Mol"""

# force
N = kg * m / s**2
"""Newton"""

# pressure
Pa = N / m**2
"""Pascal"""

kPa = 1e3 * Pa
"""Kilo-pascal"""

MPa = 1e6 * Pa
"""Mega-pascal"""

GPa = 1e6 * Pa
"""Giga-pascal"""

atm = 101325.0 * Pa
"""Standard atmosphere"""

# angle
rad = 1.0
"""Radian"""

deg = math.pi / 180.0 * rad
"""Degree"""

# frequency
Hz = 1.0
"""Heltz"""

THz = 1e12 * Hz
"""Tera-heltz"""

# Voltage
V = J / C
"""Volt"""

# Capacitance
F = C / V
"""Farad"""