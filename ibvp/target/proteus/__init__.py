"""Proteus target for IBVP translation."""

from __future__ import division

__copyright__ = "Copyright (C) 2014 Andreas Kloeckner"

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

from ibvp.language.symbolic.mappers import (
        DistributeMapper)
from ibvp.language.symbolic.util import pretty


# {{{ "transport coefficent" finding

class TransportCoefficientStorage(object):
    """"
    .. attribute:: bvp

        A :class:`ibvp.language.BVP` or a subclass.

    .. attribute:: mass

        An object array with :attr:`num_equations` entries identifying the
        term under the time derivative in each equation.

    .. attribute:: advection

        An object array of shape ``(num_equations, ambient_dim)``, where
        ``advection[i,j]`` identifies the term in the *i*th equation under
        the derivative along the *j*th axis.

    .. attribute:: diffusion

        An object array of shape
        ``(num_equations, num_equation, ambient_dim, ambient_dim)``, where
        ``diffusion[i,j,:,:]`` identifies the tensor in the *i*th equation
        multiplying the gradient of the *j*th potential.

    .. attribute:: potential

        An object array of shape ``(num_equations,)`` giving the expression
        under the second derivative ...?
        FIXME

    .. attribute:: reaction

        An object array of shape ``(num_equations,)`` giving the zeroth
        order reaction term for each equation.

    .. attribute:: hamiltonian

        An object array of shape ``(num_equations,)`` giving the expression
        depending nonlinearly on the gradients of the unknown fields in
        each equation.  These expressions do not live under a divergence.
    """

    def __init__(self, bvp):
        self.bvp = bvp

    @property
    def num_equations(self):
        return len(self.bvp.pde_system)


#class TransportCoefficientExtractor(Mapper):
#    def __init__(self, tv_storage, outers=None):
#        self.tv_storage = tv_storage
#        self.outers = outers

# }}}


def generate_proteus_problem_file(bvp):
    distr_system = DistributeMapper()(bvp.pde_system)
    for eqn in distr_system:
        print pretty(eqn)

#  vim: foldmethod=marker
