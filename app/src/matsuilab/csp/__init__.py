"""
Low-level functions for crystal structure prediction
"""
import datetime
from logging import getLogger, Logger
import math
from typing import Any, Callable, List, Sequence, Tuple, Union

import numpy as np
from matsuilab import const, unit, atpro
from numpy import ndarray


def cell_parameters(lattice:ndarray, degree:bool=False) -> Tuple[float, float, float, float, float, float]:
    """
    Returns cell parameters a, b, c, alpha, beta, gamma
    
    Args:
        lattice: lattice basis
        degree: Use degree unit if True. Otherwise, radian (default).
    """
    lattice = np.reshape(lattice, (3, 3))
    a = np.linalg.norm(lattice[0, :])
    b = np.linalg.norm(lattice[1, :])
    c = np.linalg.norm(lattice[2, :])
    alpha = math.acos(max(-1.0, min(1.0, np.dot(lattice[1, :], lattice[2, :])/(b*c))))
    beta  = math.acos(max(-1.0, min(1.0, np.dot(lattice[2, :], lattice[0, :])/(c*a))))
    gamma = math.acos(max(-1.0, min(1.0, np.dot(lattice[0, :], lattice[1, :])/(a*b))))
    if degree:
        alpha *= 180.0 / math.pi
        beta  *= 180.0 / math.pi
        gamma *= 180.0 / math.pi
    return a, b, c, alpha, beta, gamma

def cell_length(lattice:ndarray, i:int) -> float:
    """Returns cell length of a, b, or c"""
    lattice = np.reshape(lattice, (3, 3))
    return np.linalg.norm(lattice[i, :])


def cell_angle(lattice:ndarray, i:int, degree:bool=False) -> float:
    """Returns cell angles of alpha, beta, or gamma"""
    lattice = np.reshape(lattice, (3, 3))
    a = lattice[(i+1)%3, :]
    b = lattice[(i+2)%3, :]
    rad = math.acos(max(-1.0, min(1.0, np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)))))
    if degree:
        return rad * 180.0 / math.pi
    else:
        return rad


def cell_volume(lattice:ndarray, absolute:bool=True) -> float:
    """
    Returns cell volume
    
    Args:
        lattice: lattice basis
        absolute: returns absolute value if True (default)
    """
    lattice = np.reshape(lattice, (3, 3))
    if absolute:
        return abs(np.linalg.det(lattice))
    else:
        return np.linalg.det(lattice)


def reciprocal_lattice(lattice:ndarray) -> ndarray:
    """Returns reciprocal lattice basis defined with 2 \* pi \* lattice^(-1)"""
    return 2 * math.pi * np.linalg.inv(lattice.T)


def to_fractional(atoms:ndarray, lattice:ndarray) -> ndarray:
    """Returns fractional coordinates of atoms"""
    return atoms @ np.linalg.inv(lattice)


def to_cartesian(atoms:ndarray, lattice:ndarray) -> ndarray:
    """Returns cartesian coordinates of atoms"""
    return atoms @ lattice


def unique_atom_labels(numbers:Sequence[int]) -> List[str]:
    """Returns list of unique atomic labels such as [C1, C2, H1, H2]"""
    labels = []
    counts = np.zeros(100, dtype=int)
    for num in numbers:
        counts[num] += 1
        labels.append(f"{atpro.SYMBOLS[num]}{counts[num]}")
    return labels


def _join_write_return(lines, file=None, append=False) -> str:
    content = "\n".join(lines + [""])
    if file is not None:
        if append:
            with open(file, "a") as f:
                f.write(content)
        else:
            with open(file, "w") as f:
                f.write(content)
    return content


def write_cif(atoms:ndarray, lattice:ndarray,
    numbers:Sequence[int], labels:Sequence[str]=None,
    spacegroup:int=1, bonds:Sequence[Tuple[int, int, int]]=(),
    adp:ndarray=None, name:str="noname", file:str=None, append:bool=False) -> str:
    """write cif file or returns its content

    Args:
        name: name of data block
        atoms: cartesian coordinates of atoms
        lattice: cartesian coordinates of lattice vectors
        numbers: atomic numbers
        labels: atom labels, optional
        spacegroup: ID of space group
        bonds: atom indices of bonds and bond types
        adp: anisotropic displacement parameters in cartesian coordinates
        name: data name used in _data tag
        file: cif file to save
        append: whether the data is added to existing file
    """

    atoms = np.reshape(atoms, (atoms.size // 3, 3))
    lattice = np.reshape(lattice, (3, 3))

    lines = []
    lines.append(f"data_{name}")
    lines.append(f"_space_group_name_Hall           '{HALL_NAMES[spacegroup]}'")
    lines.append(f"_space_group_name_H-M_alt        '{HM_NAMES[spacegroup]}'")
    lines.append(f"_space_group_IT_number           {spacegroup}")
    lines.append(f"_cell_length_a                   {cell_length(lattice, 0):.6f}")
    lines.append(f"_cell_length_b                   {cell_length(lattice, 1):.6f}")
    lines.append(f"_cell_length_c                   {cell_length(lattice, 2):.6f}")
    lines.append(f"_cell_angle_alpha                {cell_angle(lattice, 0, True):.6f}")
    lines.append(f"_cell_angle_beta                 {cell_angle(lattice, 1, True):.6f}")
    lines.append(f"_cell_angle_gamma                {cell_angle(lattice, 2, True):.6f}")
    lines.append(f"_cell_volume                     {cell_volume(lattice):.6f}")

    lines.append(f"loop_")
    lines.append(f"_atom_site_label")
    lines.append(f"_atom_site_type_symbol")
    lines.append(f"_atom_site_fract_x")
    lines.append(f"_atom_site_fract_y")
    lines.append(f"_atom_site_fract_z")
    inv = np.linalg.inv(lattice)
    if labels is None:
        labels = unique_atom_labels(numbers)
    frac = to_fractional(atoms, lattice)
    for l, n, f in zip(labels, numbers, frac):
        lines.append(f"{l:5} {atpro.SYMBOLS[n]:2} {f[0]:.6f} {f[1]:.6f} {f[2]:.6f}")

    if adp is not None:
        lines.append(f"loop_")
        lines.append(f"_atom_site_aniso_label")
        lines.append(f"_atom_site_aniso_U_11")
        lines.append(f"_atom_site_aniso_U_12")
        lines.append(f"_atom_site_aniso_U_13")
        lines.append(f"_atom_site_aniso_U_22")
        lines.append(f"_atom_site_aniso_U_23")
        lines.append(f"_atom_site_aniso_U_33")
        nmat = np.zeros((3, 3))
        nmat[0, 0] = cell_length(inv.T, 0)
        nmat[1, 1] = cell_length(inv.T, 1)
        nmat[2, 2] = cell_length(inv.T, 2)
        inv_anmat = np.linalg.inv(nmat @ lattice)
        for l, a in zip(labels, adp):
            u = inv_anmat.T @ a @ inv_anmat
            lines.append(f"{l:5} {u[0, 0]:.6f} {u[0, 1]:.6f} {u[0, 2]:.6f} {u[1, 1]:.6f} {u[1, 2]:.6f} {u[2, 2]:.6f}")

    if len(bonds) > 0:
        lines.append(f"loop_")
        lines.append(f"_chemical_conn_bond_atom_1")
        lines.append(f"_chemical_conn_bond_atom_2")
        lines.append(f"_chemical_conn_bond_type")
        for b in bonds:
            lines.append(f"{b[0]:>4} {b[1]:>4} {CIF_BOND_TYPES[b[2]]}")

    return _join_write_return(lines, file, append)


def write_mol(name:str, atoms:ndarray, numbers:Sequence[int],
    bonds:Sequence[Tuple[int, int, int]]=(), file=None, append=False) -> str:
    """write the MDL mol file or returns its content
    
    Args:
        name: name of molecule
        atoms: atom coordinates
        numbers: list of atomic numbers
        bonds: atom indices of bonds and bond types
        file: mol file to save
        append: whether the data is added to existing file
    """

    lines = []
    lines.append(name)
    lines.append(f"  matsulab{datetime.datetime.now().strftime('%m%d%y%H%M')}3D")
    lines.append("")
    lines.append(f"{len(atoms):>3d}{len(bonds):>3d}")
    for a, n in zip(atoms, numbers):
        lines.append(f"{a[0]:>10.4f}{a[1]:>10.4f}{a[2]:>10.4f} {atpro.SYMBOLS[n]:3}")
    for b in bonds:
        lines.append(f"{b[0]:>3d}{b[1]:>3d}{b[2]:>3d}")
    lines.append("M  END")
    lines.append("$$$$")

    return _join_write_return(lines, file, append)


def line_search(fun:Callable[..., float], x0:ndarray, grad:ndarray, y0:float=None, alpha:float=1.0, maxcyc:int=10, minstep:float=1e-6, maxstep:float=1.0, logger:Logger=getLogger()) -> Tuple[ndarray, float, float, int]:
    """perform line search and returns tuple of new_x, new_y, and new_alpha"""

    logger = logger.getChild("line")

    def recalc(new_alpha, new_x1, new_y1, new_x2, new_y2):
        if new_x1 is None:
            new_x1 = x0[:] - new_alpha * grad[:]
        if new_y1 is None:
            new_y1 = fun(new_x1)
        if new_x2 is None:
            new_x2 = x0[:] - (new_alpha + new_alpha) * grad[:]
        if new_y2 is None:
            new_y2 = fun(new_x2)
        logger.debug(f"alpha= {new_alpha:12.3e}    y= {new_y1:21.12e}    max_step= {new_alpha*np.max(abs(grad)):12.3e}")
        return new_alpha, new_x1, new_y1, new_x2, new_y2

    max_abs_grad = np.max(abs(grad))

    if 2 * alpha * max_abs_grad > maxstep:
        # initial alpha is too large
        logger.info("Initial alpha is too large.")
        alpha = maxstep / (2 * max_abs_grad)
    elif alpha * max_abs_grad < minstep:
        logger.info("Initial alpha is too small.")
        alpha = minstep / max_abs_grad

    if y0 is None:
        y0 = fun(x0)
    alpha, x1, y1, x2, y2 = recalc(alpha, None, None, None, None)
    alpha3 = None
    x3 = None
    y3 = None

    cycle = maxcyc

    for i in range(maxcyc):

        # decide next action
        if y0 - y1 - y1 + y2 <= 0: # convex upwards or straight
            if y2 <= y0: # increase alpha
                if 2 * alpha * max_abs_grad < maxstep: # if below maxstep
                    logger.debug(f"Line search, cycle {i}, convec upward,   y2 optimum, increase alpha, alpha {alpha:.3g}")
                    alpha, x1, y1, x2, y2 = recalc(alpha+alpha, x2, y2, None, None)
                else: # if maxstep
                    logger.debug(f"Line search, cycle {i}, convec upward,   y2 optimum, max step,       alpha {alpha:.3g}")
                    alpha, x1, y1, x2, y2 = recalc(maxstep/(2*max_abs_grad), None, None, None, None)
                    cycle = i+1
                    break
            else: # decrease alpha
                logger.debug(f"Line search, cycle {i}, convec upward,   y0 optimum, decrease alpha, alpha {alpha:.3g}")
                alpha, x1, y1, x2, y2 = recalc(alpha*0.5, None, None, x1, y1)
        else: # convex downwards
            alpha3 = alpha * 0.5 * (3 * y0 - 4 * y1 + y2) / (y0 - 2 * y1 + y2)
            x3 = x0[:] - alpha3 * grad[:]
            y3 = fun(x3)
            if y3 <= min(y0, y1, y2):
                if alpha3 * max_abs_grad < maxstep:
                    logger.debug(f"Line search, cycle {i}, convec downward, y3 optimum,                 alpha {alpha:.3g}")
                    cycle = i+1
                    break
                else:
                    logger.debug(f"Line search, cycle {i}, convec downward, y3 optimum, max step,       alpha {alpha:.3g}")
                    alpha, x1, y1, x2, y2 = recalc(maxstep/(2*max_abs_grad), None, None, None, None)
                    x3, y3 = None, None
                    cycle = i+1
                    break
            elif y1 <= min(y0, y2):
                logger.debug(f"Line search, cycle {i}, convec downward, y1 optimum,                 alpha {alpha:.3g}")
                cycle = i+1
                break
            elif y0 <= y2:
                logger.debug(f"Line search, cycle {i}, convec downward, y0 optimum, decrease alpha, alpha {alpha:.3g}")
                alpha, x1, y1, x2, y2 = recalc(alpha*0.5, None, None, x1, y1)
            else:
                if 2 * alpha * max_abs_grad < maxstep: # if below maxstep
                    logger.debug(f"Line search, cycle {i}, convec downward, y2 optimum, increase alpha, alpha {alpha:.3g}")
                    alpha, x1, y1, x2, y2 = recalc(alpha+alpha, x2, y2, None, None)
                else: # if maxstep
                    logger.debug(f"Line search, cycle {i}, convec downward, y2 optimum, max step,       alpha {alpha:.3g}")
                    alpha, x1, y1, x2, y2 = recalc(maxstep/(2*max_abs_grad), None, None, None, None)
                    cycle = i+1
                    break
    
    if cycle == maxcyc:
        logger.info(f"Line search has reached maximum cycles: {cycle}")

    if y3 is not None and y3 < min(y1, y2):
        # logger.info(f"y3 is optimum in line search cycle {cycle}")
        return x3, y3, alpha3, cycle
    elif y1 < y2:
        # logger.info(f"y1 is optimum in line search cycle {cycle}")
        return x1, y1, alpha, cycle
    else:
        # logger.info(f"y2 is optimum in line search cycle {cycle}")
        return x2, y2, alpha+alpha, cycle


class DictLike(dict):
    """Super class of dict. Items can be accessed as attributes. For example, d.x instead of d["x"]"""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            return None
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, kwargs=None):
        if kwargs is not None:
            dict.__init__(self, **kwargs)


def minimize(fun:Callable[..., float], x0:ndarray,
    mask:ndarray=None,
    gfun:Callable[..., ndarray]=1e-6,
    hfun:Callable[..., ndarray]=None, conjugate:bool=True,
    maxcyc:int=1000, alpha0:float=1.0, maxstep:float=1.0, linecyc:int=10,
    criteria:Sequence[Callable[..., bool]]=None,
    callback:Sequence[Callable[..., bool]]=(), logger:Logger=getLogger()) -> DictLike:
    """Find x which minimizes fun(x)
    
    Args:
        fun: The objective function to be minimized.
        x0: initial guess
        mask: array of bool. True for fixing varialbes in x0
        gfun: gradient function. may be delta (float) for numerical differentiation
        hfun: hessian function for modified newton method.
        conjugate: whether conjugate gradient method in Polak–Ribiere's form is used
        maxcyc: maximum number of cycles
        alpha0: initial guess of step in line search. x[:] is updated as x[:] -= alpha \* grad[:]
        maxstep: upper limit of single step length
        linecyc: maximum number of cycles for line search. If linecyc is 0 or None, the given alpha is used as is.
        criteria: convergence criteria. callable or list of callables
        callback: called before the first iteration and after every iteration.
    
    Returns:
        DictLike: dict-like object including all local variables
    """

    logger = logger.getChild("min")

    # set arguments if None
    if type(gfun) is float:
        if gfun > 0:
            gfun = numerical_derivative(fun=fun, shape_in=x0.size, mask=mask, delta=gfun, reuse=True) 
        else:
            gfun = numerical_derivative(fun=fun, shape_in=x0.size, mask=mask, delta=-gfun, reuse=True, forward=True) 
    if hfun is not None:
        conjugate = False
    if mask is None:
        mask = np.full(len(x0), False, dtype=bool)
    not_mask = np.logical_not(mask)
    if criteria is None:
        criteria = (y_criteria(), x_criteria(), g_criteria(), h_criteria())
    if callback is None:
        callback = ()

    # declare variables
    x = np.array(x0, copy=False)
    x = x.reshape(x.size)
    old_x = np.zeros(x.shape)
    y = None
    mask = mask
    alpha = alpha0
    cycle = 0
    newton = False
    successful = False
    failed = False
    if conjugate:
        old_grad = np.zeros(x.shape)
    else:
        old_grad = None
    if conjugate or hfun is not None:
        dire = np.zeros(x.shape)
    else:
        dire = None

    refreshed = any(_funmap(callback, DictLike(locals())))

    for cycle in range(1, maxcyc+1):

        # update gradient
        if conjugate and cycle > 1:
            old_grad[:] = grad[:]

        # if type(gfun) is float:
        #     grad = numerical_gradient(fun=fun, x=x, mask=mask, grad=grad, delta=gfun)
        # else:
        grad = gfun(x)
        grad[mask] = 0.0

        # decide search direction
        if hfun is not None:
            # modified Newton method
            hess = hfun(x)
            eigenvalues, eigenvectors = np.linalg.eigh(hess[not_mask, :][:, not_mask])
            non_positive = (eigenvalues <= 0).astype(int)
            if np.count_nonzero(non_positive) == 0:
                logger.debug(f"Hessian is positive difinite. cycle {cycle}")
                newton = True
                dire[not_mask] = ((grad[not_mask] @ eigenvectors) / eigenvalues) @ eigenvectors.T
            else:
                logger.info(f"Hessian is not positive difinite. cycle {cycle}, count {np.count_nonzero(non_positive)}")
                newton = False
                dire[not_mask] = ((grad[not_mask] @ eigenvectors) * non_positive) @ eigenvectors.T
        elif conjugate:
            # conjugate gradient method
            if cycle == 1:
                dire[:] = grad
            else:
                # conjugate gradient (Polak–Ribiere's form)
                dot_old_old = np.dot(old_grad, old_grad)
                if dot_old_old == 0.0:
                    successful = True
                    break
                else:
                    beta = max(0.0, np.dot(grad, grad - old_grad) / dot_old_old)
                    dire[:] = grad + beta * dire
        else:
            # gradient descent method
            dire = grad

        # update x, y, alpha
        if y is None:
            y = fun(x)
        old_x[:] = x[:]
        old_y = y
        if linecyc is None or linecyc == 0:
            alpha = min(alpha, maxstep / np.max(abs(dire)))
            x[:] -= alpha * dire[:]
            y = fun(x)
        else:
            if newton:
                x[:], y, _, cycle2 = line_search(fun, x, dire, None, 1.0, linecyc, maxstep, logger=logger)
            else:
                x[:], y, alpha, cycle2 = line_search(fun, x, dire, None, alpha, linecyc, maxstep, logger=logger)

        # check convergence criteria
        successful = all(_funmap(criteria, DictLike(locals())))
        if not successful and cycle == maxcyc:
            faield = True

        refreshed = any(_funmap(callback, DictLike(locals())))

        if successful:
            break

    return DictLike(locals())


def numerical_derivative(fun:Callable[[ndarray], ndarray], shape_in:Tuple[int, ...]=(), shape_out:Tuple[int, ...]=(), mask:ndarray=None, delta:float=1e-3, order:int=1, all:bool=False, reuse:bool=False, forward:bool=False) -> Callable[[ndarray], ndarray]:
    """Returns a numerical derivative function.
    
    Args:
        fun: function
        shape_in: shape of input array
        shape_out: shape of output array
        mask: mask for excluding calculations of the specified dimensions
        delta: small change in numerical differentiation
        order: returns *order*-th derivative
        all: whether it returns a list of all derivatives up to *order*-th derivatives
        reuse: whether the returned arrays of multiple function calls shares memory

    Returns:
        Callable[[ndarray], ndarray]
    """
    if type(shape_in) is int:
        shape_in = (shape_in,)
    else:
        shape_in = tuple(shape_in)
    if type(shape_out) is int:
        shape_out = (shape_out,)
    else:
        shape_out = tuple(shape_out)
    size_in = np.prod(shape_in)
    der = np.zeros(shape_out + shape_in)
    if mask is None:
        mask = np.full(size_in, False, dtype=bool)
    def der_fun(x:ndarray) -> ndarray:
        x = np.reshape(x, size_in)
        if len(shape_out) == 0:
            d = np.reshape(der, (1,) + (size_in,))
        else:
            d = np.reshape(der, shape_out + (size_in,))
        if forward:
            y0 = fun(x)
            for i in range(size_in):
                if not mask[i]:
                    x0 = x[i]
                    x[i] = x0 + delta
                    d[:, i] = (fun(x) - y0) / delta
                    x[i] = x0
        else:
            for i in range(size_in):
                if not mask[i]:
                    x0 = x[i]
                    x[i] = x0 - delta
                    d[:, i] = fun(x)
                    x[i] = x0 + delta
                    d[:, i] -= fun(x)
                    x[i] = x0
                    d[:, i] /= delta + delta
        if reuse:
            return der
        else:
            return np.array(der)
    if order < 1:
        raise Exception("order must be positive integer.")
    elif order == 1:
        if all:
            return [der_fun]
        else:
            return der_fun
    else:
        if all:
            return [der_fun] + numerical_derivative(fun=der_fun, shape_in=shape_in, shape_out=shape_out+shape_in, mask=mask, delta=delta, order=order-1, all=True)
        else:
            return numerical_derivative(fun=der_fun, shape_in=shape_in, shape_out=shape_out+shape_in, mask=mask, delta=delta, order=order-1)


def y_criteria(tol:float=1e-9, count:int=3) -> Callable[[DictLike], bool]:
    """
    Returns a criteria function for the tolerance of delta y. Useful in *minimize* function.
    """
    if tol is None:
        return lambda o: True
    d = DictLike()
    d.score = 0
    def criteria(o:dict):
        if d.old_y is not None and abs(d.old_y - o.y) <= tol:
            d.score += 1
            if d.score >= count:
                return True
        else:
            d.score = 0
        d.old_y = o.y
        return False
    return criteria


def x_criteria(tol:float=1e-3, count:int=3) -> Callable[[DictLike], bool]:
    """
    Returns a criteria function for the tolerance of delta x. Useful in *minimize* function.
        
    Returns:
        Callable[[DictLike], bool]
    """
    if tol is None:
        return lambda o: True
    d = DictLike()
    d.score = 0
    def criteria(o:dict):
        if d.old_x is not None and np.linalg.norm((d.old_x - o.x)[np.logical_not(o.mask)]) < tol:
            d.score += 1
            if d.score >= count:
                return True
        else:
            d.score = 0
        if d.old_x is None:
            d.old_x = np.empty(o.x.shape)
        d.old_x[:] = o.x[:]
        return False
    return criteria


def g_criteria(tol:float=1e-3, count:int=3) -> Callable[[DictLike], bool]:
    """
    Returns a criteria function for the tolerance of gradient. Useful in *minimize* function.
        
    Returns:
        Callable[[DictLike], bool]
    """
    if tol is None:
        return lambda o: True
    d = DictLike()
    d.score = 0
    def criteria(o:dict):
        if np.linalg.norm(o.grad[np.logical_not(o.mask)]) < tol:
            d.score += 1
            if d.score >= count:
                return True
        else:
            d.score = 0
        return False
    return criteria


def h_criteria() -> Callable[[DictLike], bool]:
    """
    Returns a criteria function for positive definite hessian. Useful in *minimize* function.
    """
    def criteria(o:DictLike):
        if o.eigenvalues is None:
            return True
        else:
            return all(o.eigenvalues > 0.0)
    return criteria


def default_report(interval:int=1, header:str=None) -> Callable[[DictLike], bool]:
    """
    Returns a simple report function for callback in *minimize* function.
    """
    if interval < 1:
        return lambda o: False
    if header is None:
        header = "cycle                  y      delta_y    norm_grad norm_delta_x        alpha"
    def report(o:DictLike):
        if o.cycle == 0:
            print(header)
            if o.y is None:
                o.y = o.fun(o.x)
            if o.grad is None:
                o.grad = o.gfun(o.x)
            print(f"{o.cycle:5} {o.y:18.9e} {0.0:12.3e} {np.linalg.norm(o.grad[np.logical_not(o.mask)]):12.3e} {0.0:12.3e} {0.0:12.3e}")
        elif o.cycle % interval == 0 or o.successful:
            norm_delta_x = np.linalg.norm(o.x - o.old_x)
            delta_y = o.old_y - o.y
            print(f"{o.cycle:5} {o.y:18.9e} {delta_y:12.3e} {np.linalg.norm(o.grad[np.logical_not(o.mask)]):12.3e} {norm_delta_x:12.3e} {o.alpha:12.3e}")
        if o.successful:
            print("Optimization completed.")
        elif o.failed:
            print("Optimization not completed yet.")
        # o.old_x[:] = o.x
        # o.old_y = o.y
        return False
    return report


def _funmap(fun, *args, **kwargs):
    """call multiple functions"""
    if callable(fun):
        fun = [fun]
    return [f(*args, **kwargs) for f in fun]


def minimize_free_energy(atoms:ndarray, lattice:ndarray, temp:float,
    numbers:ndarray, labels:Sequence[str],
    fun:Callable[..., float], gfun:Callable, hfun:Callable=None,
    delta:float=1e-4, conjugate:bool=False,
    maxcyc:int=1000, xtol:float=1e-3, ytol:float=1e-9, gtol:float=1e-3,
    alpha0:float=1.0, maxstep:float=1.0, linecyc:int=10,
    irefresh:int=10, refresh:Callable=None,
    ireport:int=1, report:Callable=None, trajectory:str=None):
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

    n = atoms.size // 3
    atoms = np.reshape(atoms, atoms.size)
    lattice = np.reshape(lattice, lattice.size)
    x0 = np.array(lattice[(0, 3, 4, 6, 7, 8)])

    mask = np.array(tuple([i < 3 for i in range(atoms.size)]) + ((True,)*9), dtype=bool)

    hessian = np.zeros((n+3, 3, n+3, 3))
    hess11 = np.zeros((n, 3, n, 3))
    hess12 = np.zeros((n, 3, 3, 3))
    hess22 = np.zeros((3, 3, 3, 3))

    sub_atoms = np.array(atoms)
    sub_lattice = np.array(lattice)

    def fun(x):
        sub_lattice[(0, 3, 4, 6, 7, 8)] = x[:]
        result = minimize_lattice_energy(sub_atoms, sub_lattice, numbers=numbers, labels=labels,
            fun=fun, gfun=gfun, hfun=hfun, delta=delta, conjugate=conjugate, mask=mask,
            maxcyc=maxcyc, xtol=xtol, ytol=ytol, gtol=gtol,
            alpha0=alpha0, maxstep=maxstep, linecyc=linecyc,
            irefresh=0, refresh=None, ireport=0, report=None, trajectory=None)
        lattice_energy = fun(result.x[:-9], result.x[-9:])
        hfun(result.x[:-9], result.x[-9:], hess11, hess12, hess22, delta)
        frequencies, _ = calc_phonons(mass=atpro.WEIGHTS[numbers], hess11=hess11, temp=temp, eunit="kcal/mol")
        phonon_energy = phonon_free_energy(frequencies, temp)
        return lattice_energy + phonon_energy

    def ref(o:DictLike):
        if irefresh > 0 and o.cycle % irefresh == 0:
            refresh(o.x[:-9], o.x[-9:])
            sub_lattice[(0, 3, 4, 6, 7, 8)] = o.x[:]
            result = minimize_lattice_energy(sub_atoms, sub_lattice, numbers=numbers, labels=labels,
                fun=fun, gfun=gfun, hfun=hfun, delta=delta, conjugate=conjugate, mask=mask,
                maxcyc=maxcyc, xtol=xtol, ytol=ytol, gtol=gtol,
                alpha0=alpha0, maxstep=maxstep, linecyc=linecyc,
                irefresh=0, refresh=None, ireport=0, report=None, trajectory=None)
            sub_atoms[:] = result.x[:-9]
            return True
        else:
            return False

    if ireport > 0:
        if report is None:
            print()
            print("==== Free Energy Minimization ====")
            print("cycle             energy      delta_E    norm_grad norm_delta_x        alpha")

            def rep(o:DictLike):
                if o.cycle % ireport == 0 or o.successful:
                    if o.cycle > 0:
                        norm_delta_x = np.linalg.norm(o.x - o.old_x)
                        delta_e = o.old_y - o.y
                    else:
                        norm_delta_x = 0.0
                        delta_e = 0.0
                    print(f"{o.cycle:5} {o.y:18.9e} {delta_e:12.3e} {np.linalg.norm(o.grad[np.logical_not(mask)]):12.3e} {norm_delta_x:12.3e} {o.alpha:12.3e}")
                    if trajectory is not None:
                        write_cif(name=str(o.cycle), atoms=o.x[:-9], lattice=o.x[-9:],
                                    numbers=numbers, labels=labels,
                                    spacegroup=1, file=trajectory, append=True)
        else:
            def rep(o:DictLike):
                if o.cycle % ireport == 0 or o.successful:
                    report(o)
    else:
        rep = None

    result = minimize(fun=fun, x0=x0, mask=None, gfun=delta, hess=None,
        conjugate=True, maxcyc=maxcyc, xtol=xtol, ytol=ytol, gtol=gtol,
        alpha0=alpha0, maxstep=maxstep, linecyc=linecyc,
        refresh=ref, report=rep)

    atoms[:] = sub_atoms[:]
    lattice[(0, 3, 4, 6, 7, 8)] = result.x[:]

    if ireport > 0 and report is None:
        if result.successful:
            print("Optimization completed.")
        else:
            print("Optimization not completed yet.")

    return result


def calc_phonons(mass:ndarray, hess11:ndarray, temp:float, eunit:str="kcal/mol", logger:Logger=getLogger()) -> Tuple[ndarray, ndarray]:
    """Returns phonon frequencies and modes.
    
    Args:
        mass: list of atomic mass in Da
        hess11: hessian
        temp: temperature
        eunit: unit of energy in hessian
    """

    logger = logger.getChild("pho")

    mass = mass * unit.Da
    inv_sqrt_mass = 1.0 / np.sqrt(mass)

    n = len(mass)
    dmat = np.array(hess11)
    for i in range(n):
        for j in range(n):
            dmat[i, :, j, :] *= inv_sqrt_mass[i] * inv_sqrt_mass[j]
    dmat = np.reshape(dmat, (n*3, n*3))

    # eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(dmat)
    eigenvectors = eigenvectors.T

    # phonon frequencies squared ((rad/s)**2)
    eigenvalues = convert_energy_unit(eigenvalues, eunit, "J") / (unit.kg * unit.Angstrom**2)

    # remove three translation modes and check imaginary phonons
    positive = eigenvalues > 1e24
    if np.count_nonzero(positive) != 3*n-3:
        logger.warning(f"The number of phonons invalid: expected {3*n-3}, found {np.count_nonzero(positive)}")
        # print(f"omega squared:")
        # print(f"{eigenvalues}")
    frequencies = np.sqrt(eigenvalues[positive])
    modes = np.reshape(eigenvectors, (n*3, n, 3))[positive, :, :]

    # phonon modes
    for mode, omega in zip(modes, frequencies):
        if temp == 0:
          pop = (const.k_b * temp) / (const.hbar * omega)
        else:
          pop = 1.0 / (math.exp(const.hbar * omega / (const.k_b * temp)) - 1.0)
        for m, xyz in zip(mass, mode):
            xyz[:] *= math.sqrt(const.hbar / (2 * m * omega) * (1.0 + 2.0 * pop)) / unit.Angstrom

    # unit conversion to Hz
    frequencies *= 1.0 / (2.0 * math.pi)

    return frequencies, modes


def anisotropic_displacement_parameters(modes:ndarray) -> ndarray:
    adp = np.zeros((modes.shape[1], 3, 3))
    for m in modes: # loop for the number of phonon modes
        for a, v in zip(adp, m): # loop for the number of atoms
            a += np.outer(v, v)
    return adp


def phonon_free_energy(frequencies:ndarray, temp:float, eunit="kcal/mol") -> float:
    """frequencies (Hz), temperature (K) -> phonon_free_energy (kcal/mol)"""
    if temp == 0:
        e = np.sum(0.5 * const.h * frequencies)
    else:
        kbT = const.k_b * temp
        e = np.sum(kbT * np.log(2.0 * np.sinh(const.h / (2.0 * kbT) * frequencies)))
    return convert_energy_unit(e, "J", eunit)


def convert_energy_unit(e:float, fr:str="J", to:str="J") -> float:
    if fr == "kcal/mol":
        e *= unit.kcal / unit.mol
    elif fr == "kJ/mol":
        e *= unit.kJ / unit.mol
    elif fr == "eV":
        e *= unit.eV
    elif fr == "J":
        e *= unit.J
    else:
        raise NotImplementedError(f"unknown unit: {fr}")

    if to == "kcal/mol":
        e /= unit.kcal / unit.mol
    elif to == "kJ/mol":
        e /= unit.kJ / unit.mol
    elif to == "eV":
        e /= unit.eV
    elif to == "J":
        e /= unit.J
    else:
        raise NotImplementedError(f"unknown unit: {to}")

    return e


def adjacency_list_to_adjacency_matrix(adjlist:Sequence[Tuple[int, int, Any]], n:int, i0:int=1) -> ndarray:
    """Convert adjacency list to adjacency matrix. i0 is the start of index (0 or 1)."""
    if len(adjlist[0]) == 2:
        matrix = np.zeros((n, n), dtype=int)
        for a in adjlist:
            matrix[a[0] - i0, a[1] - i0] = 1
            matrix[a[1] - i0, a[0] - i0] = 1
    elif len(adjlist[0]) == 3:
        matrix = np.zeros((n, n), dtype=type(adjlist[0][2]))
        for a in adjlist:
            matrix[a[0] - i0, a[1] - i0] = a[2]
            matrix[a[1] - i0, a[0] - i0] = a[2]
    else:
        raise Exception("Size of adjacency list is invalid.")
    return matrix


def adjacency_matrix_to_distance_matrix(adjmat:ndarray=None) -> ndarray:
    n = adjmat.shape[0]
    adjmat = adjmat.astype(bool)
    dist = np.full(adjmat.shape, n, dtype=int)
    for i in range(n):
        dist[i, i] = 0
    for _ in range(n):
        for d in dist:
            for i, a in enumerate(adjmat):
                if np.count_nonzero(a) > 0:
                    d[i] = min(np.min(d[a]) + 1, d[i])
    for d in dist:
        for i in range(n):
            if d[i] == n:
                d[i] = -1
    return dist


def distance_matrix_to_exclusion_matrix(dist:ndarray, elist:Sequence[float]) -> ndarray:
    n = dist.shape[0]
    exc = np.zeros(dist.shape)
    for i in range(n):
        for j in range(n):
            if dist[i, j] < 0 or dist[i, j] >= len(elist):
                exc[i, j] = 0.0
            else:
                exc[i, j] = elist[dist[i, j]]
    return exc


CIF_BOND_TYPES = ("", "sing", "doub", "trip", "arom")
"""bond types used for _chemical_conn_bond_type tag of cif file"""


HM_NAMES = ('', 'P 1', 'P -1', 'P 1 2 1', 'P 1 21 1', 'C 1 2 1', 'P 1 m 1', 'P 1 c 1', 'C 1 m 1', 'C 1 c 1', 'P 1 2/m 1',
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
"""Hermann–Mauguin names of space group used in cif"""


HALL_NAMES = ('', 'P 1', '-P 1', 'P 2y', 'P 2yb', 'C 2y', 'P -2y', 'P -2yc', 'C -2y', 'C -2yc', '-P 2y',
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
"""Hall names of space group used in cif"""
