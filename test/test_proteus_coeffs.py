"""Proteus target for IBVP translation."""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

__copyright__ = "Copyright (C) 2014 Rob Kirby, Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import numpy as np

import ibvp.sym as sym
from ibvp.language import PDESystem
from ibvp.target.proteus import generate_proteus_problem_file


# {{{ advection

def test_advection():
    u = sym.Field("u")

    eqns = sym.join(sym.d_dt(u) + sym.d_dx(u))

    print(sym.pretty(eqns))

    print(generate_proteus_problem_file(
            PDESystem(
                ambient_dim=1,
                pde_system=eqns,
                unknowns=[u],
                ),
            "Advection"))

# }}}


# {{{ burgers

def test_burgers():
    ambient_dim = 2

    u = sym.Field("u")

    vec = np.array([1.0, 2.0])

    eqns = sym.join(
            sym.d_dt(u)
            + sym.div(vec * u**2)
            - sym.div(sym.grad(u))
            )

    print(sym.pretty(eqns))

    generate_proteus_problem_file(
            PDESystem(
                ambient_dim=ambient_dim,
                pde_system=eqns,
                unknowns=[u],
                ),
            "Burgers")

# }}}


# {{{ heat equation

def test_heat():
    ambient_dim = 2

    u = sym.Field("u")

    eqns = sym.join(
            sym.d_dt(u)
            - sym.div(sym.grad(u))
            )

    print(sym.pretty(eqns))

    generate_proteus_problem_file(
            PDESystem(
                ambient_dim=ambient_dim,
                pde_system=eqns,
                unknowns=[u],
                ),
            "Heat")

# }}}


# {{{ parabolic system

def test_parabolic_system():
    ambient_dim = 2

    u = sym.Field("u")
    v = sym.Field("v")

    eqns = sym.join(
            sym.d_dt(u) - sym.div(sym.grad(u-v)),
            sym.d_dt(v) - sym.div(sym.grad(u+v)),
            )

    print(sym.pretty(eqns))

    generate_proteus_problem_file(
            PDESystem(
                ambient_dim=ambient_dim,
                pde_system=eqns,
                unknowns=[u, v],
                ),
            "ParabolicSystem")

# }}}


# {{{ 2nd order wave equation

def test_wave():
    dim = 3
    u = sym.Field("u")
    f = sym.Field("f")
    c = sym.Parameter("c")
    v = sym.VectorField("v")

    eqns = sym.join(
            sym.d_dt(u)
            + c * sym.div(v) - f,

            sym.d_dt(v)
            + c * sym.grad(u)
            )

    print(sym.pretty(eqns))

    generate_proteus_problem_file(
            PDESystem(
                ambient_dim=dim,
                pde_system=eqns,
                unknowns=[u, v],
                ),
            "Wave")

# }}}


# {{{ 'very nonlinear burgers'

def test_very_nonlinear_burgers():
    ambient_dim = 2

    p = 1
    q = 2
    r = 2

    u = sym.Field("u")

    B = np.array([1.0, 2.0])

    Adiff = 0.001
    C = 0.0001

    eqns = sym.join(
            sym.d_dt(u**p)
            + sym.div(B * u**q)
            - sym.div(Adiff * sym.grad(u**r))
            + C * u
            )

    print(sym.pretty(eqns))

    print(generate_proteus_problem_file(
            PDESystem(
                ambient_dim=ambient_dim,
                pde_system=eqns,
                unknowns=[u],
                ),
            "Burgers"))

# }}}


# {{{ shallow water (using geometric calculus)

def test_shallow_water_gc():
    import ibvp.sym as sym

    dim = 2
    col_height = sym.Field("eta")

    # "mom"entum really is column height * velocity
    mom = sym.MultiVectorField("mom")

    d1 = sym.Derivative()
    d2 = sym.Derivative()
    d3 = sym.Derivative()
    eqns = sym.join(
            sym.d_dt(col_height)
            + (d1.nabla | d1(mom)),

            sym.d_dt(mom)
            + d2(mom * (mom | d2.nabla)/col_height)
            + 0.5*d3.nabla*d3(col_height**2)
            )

    print(sym.pretty(eqns))

    from ibvp.language import PDESystem
    system = PDESystem(
            ambient_dim=dim,
            pde_system=eqns,
            unknowns=[col_height, mom],
            )

    from ibvp.target.proteus import generate_proteus_problem_file
    generate_proteus_problem_file(system, "ShallowWater")

# }}}


# You can test individual routines by typing
# $ python test_proteus_coeffs.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: foldmethod=marker
