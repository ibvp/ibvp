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

import numpy as np
from ibvp.language.symbolic.mappers import (
        Scalarizer, DistributeMapper, CombineMapper, WalkMapper)
from ibvp.language.symbolic.util import pretty
import ibvp.language.symbolic.primitives as p
import pymbolic.primitives as pp


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

    def __init__(self, bvp, scalar_unknowns):
        self.bvp = bvp
        self.scalar_unknowns = scalar_unknowns

        adim = bvp.ambient_dim
        neq = self.num_equations

        self.mass = np.zeros(neq, dtype=object)
        self.advection = np.zeros((neq, adim),
                dtype=object)
        self.diffusion = np.zeros((neq, neq, adim, adim), dtype=object)
        self.potential = np.zeros(neq, dtype=object)
        self.reaction = np.zeros(neq, dtype=object)
        self.hamiltonian = np.zeros(neq, dtype=object)

    @property
    def num_equations(self):
        return len(self.scalar_unknowns)


#class TransportCoefficientExtractor(Mapper):
#    def __init__(self, tv_storage, outers=None):
#        self.tv_storage = tv_storage
#        self.outers = outers

# }}}


def pick_off_constants(expr):
    if isinstance(expr, pp.Product):
        constants = []
        non_constants = []

        for child in expr.children:
            if isinstance(child, pp.Product):
                sub_const, sub_expr = pick_off_constants(child)
                constants.append(sub_const)
                non_constants.append(sub_expr)
            elif pp.is_constant(child) or isinstance(child, p.Parameter):
                constants.append(child)
            else:
                non_constants.append(child)

        return (pp.flattened_product(constants),
                pp.flattened_product(non_constants))

    else:
        return 1, expr


# {{{ supporting mappers

class HasSomethingMapper(CombineMapper):
    def combine(self, values):
        import operator
        return reduce(operator.or_, values)

    def map_constant(self, expr):
        return False

    map_field = map_constant
    map_derivative = map_constant
    map_time_derivative = map_constant


class HasTimeDerivativeMapper(HasSomethingMapper):
    def map_time_derivative(self, expr):
        return True


class HasSpatialDerivativeMapper(HasSomethingMapper):
    def map_derivative(self, expr):
        return True

    map_div = map_derivative
    map_grad = map_derivative
    map_curl = map_derivative


class MaxSubscriptFinder(WalkMapper):
    def __init__(self):
        self.name_to_max_index = {}

    def map_subscript(self, expr):
        if isinstance(expr.aggregate, p.Field):
            assert isinstance(expr.index, int)
            self.name_to_max_index[expr.aggregate.name] = max(
                    self.name_to_max_index.get(expr.aggregate.name, 0),
                    expr.index)

# }}}


def generate_proteus_problem_file(bvp):
    scalarized_system = Scalarizer(bvp.ambient_dim)(bvp.pde_system)

    import ibvp.sym as sym
    print sym.pretty(scalarized_system)

    distr_system = DistributeMapper()(scalarized_system)

    msf = MaxSubscriptFinder()
    msf(distr_system)

    scalar_unknowns = []
    for unk in bvp.unknowns:
        unk_max_index = msf.name_to_max_index.get(unk)
        if unk_max_index is not None:
            scalar_unknowns.extend(
                    "%s_%d" % (unk, i)
                    for i in range(unk_max_index+1))
        else:
            scalar_unknowns.append(unk)

    if len(set(scalar_unknowns)) != len(scalar_unknowns):
        raise ValueError("names of unknowns not unique "
                "after scalarization")

    import ibvp.sym as sym
    print sym.pretty(distr_system)

    tc_storage = TransportCoefficientStorage(bvp, scalar_unknowns)

    has_time_derivative = HasTimeDerivativeMapper()
    has_spatial_derivative = HasSpatialDerivativeMapper()

    for i, eqn_i in enumerate(distr_system):
        if isinstance(eqn_i, pp.Sum):
            terms = eqn_i.children
        else:
            terms = (eqn_i,)

        for term in terms:
            constant, term_without_constant = pick_off_constants(term)

            if isinstance(term_without_constant, p.OperatorBinding):
                op = term_without_constant.op

                if isinstance(op, p.TimeDerivativeOperator):
                    if has_spatial_derivative(term_without_constant.argument):
                        raise ValueError("no spatial derivatives allowed inside "
                                "of time derivative")
                    tc_storage.mass[i] += (
                            constant * term_without_constant.argument)
                    continue

                if has_time_derivative(term_without_constant):
                    raise ValueError("time derivatives found below "
                            "root of tree of term '%s'" % pretty(term))

                if isinstance(op, p.DerivativeOperator):
                    outer_deriv_axis = term_without_constant.op.ambient_axis
                    outer_deriv_argument = term_without_constant.argument

                    if not has_spatial_derivative(outer_deriv_argument):
                        tc_storage.advection[i, outer_deriv_axis] += (
                                constant * outer_deriv_argument)
                    else:
                        raise NotImplementedError()
                else:
                    raise ValueError("unexpected operator: %s"
                            % type(term_without_constant.op).__name__)
            else:
                if has_time_derivative(term_without_constant):
                    raise ValueError("time derivatives found below "
                            "root of tree of term '%s'" % pretty(term))

                if has_spatial_derivative(term_without_constant):
                    tc_storage.hamiltonian[i] += term
                else:
                    tc_storage.reaction[i] += term




#  vim: foldmethod=marker