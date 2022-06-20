import time

from matsuilab.csp.forcefield import ForceField
from matsuilab.csp.optcrystal import OptCrystal


def main(*args):
    if len(args) < 4:
        print("usage: python -m matsuilab.csp.lem input.ff input.cif output.cif 1")
    else:
        fffile = args[0]
        input = args[1]
        output = args[2]
        opt_lattice = str2bool(args[3])

        ff = ForceField.open(fffile)
        crystal = OptCrystal.read_cif(input)

        start = time.time()
        crystal.minimize_lattice_energy(ff, opt_lattice=opt_lattice, trajectory=output)
        end = time.time()

        print(f"Elapsed Time: {end-start} s")


def str2bool(s:str):
    if s[0] == "F" or s[0] == "f" or s[0] == "0":
        return False
    else:
        return True


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
