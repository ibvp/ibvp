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
from pymbolic.mapper.dependency import DependencyMapper as DepMapBase
from pymbolic.mapper.differentiator import DifferentiationMapper as DBase
import pymbolic as pmbl


class DependencyMapper(DepMapBase):
    def map_field(self, expr):
        return set([expr])

    def map_parameter(self, expr):
        return set([expr])

    def map_operator_binding(self, expr):
        return self.combine([
            self.rec(expr.op), self.rec(expr.argument)])


def classify_dep(expr):
    if expr:
        deptype = 'linear'
        dmapper = DependencyMapper()
        deps = dmapper(expr)
        if deps:
            for dep in deps:
                if isinstance(dep, p.Field):
                    deptype = 'nonlinear'
                    break
    else:
        deptype = 'constant'

    return deptype


class DMapper(DBase):
    map_field = pmbl.mapper.differentiator.DifferentiationMapper.map_variable


def differentiate(expr, var):
    return DMapper(var)(expr)


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

        self.potential_registry = {}

    def __str__(self):
        import ibvp.sym as sym

        def dump_1d(title, thing):
            mystr = 'Nonzero %s terms\n-----------------------\n' % (title,)
            for i, m in enumerate(thing):
                if m:
                    mystr += "%10d: %s\n" % (i, sym.pretty(m))
            mystr += '\n'
            return mystr

        mystr = dump_1d("Mass", self.mass)
        mystr += 'Nonzero Advection terms\n-----------------------\n'
        for i, bi in enumerate(self.advection):
            for j, bij in enumerate(bi):
                if bij:
                    mystr += "%5d%5d: %s\n" % (i, j, sym.pretty(bij))
        mystr += '\n'

        mystr += 'Nonzero Diffusion coefficients\n-------------------------\n'
        for i, ai in enumerate(self.diffusion):
            for j, aij in enumerate(ai):
                for k, aijk in enumerate(aij):
                    for ell, aijkell in enumerate(aijk):
                        if aijkell:
                            mystr += ("%5d%5d%5d%5d: %s\n"
                                    % (i, j, k, ell, sym.pretty(aijkell)))

        mystr += '\n'

        mystr += dump_1d("Potential", self.potential)

        mystr += dump_1d("Reaction", self.reaction)
        mystr += dump_1d("Hamiltonian", self.hamiltonian)

        return mystr

    @property
    def num_equations(self):
        return len(self.scalar_unknowns)

    def register_potential(self, expr):
        try:
            return self.potential_registry[expr]
        except KeyError:
            if len(self.potential_registry) >= self.num_equations:
                # FIXME: Really?
                raise ValueError("number of potentials exhausted")

            i = len(self.potential_registry)
            self.potential[i] = expr
            self.potential_registry[expr] = i
            return i


def pick_off_constants(expr):
    """
    :return: a tuple ``(constant, non_constant)`` that contains
        separates out nodes constant multipliers from any other
        nodes in *expr*
    """

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


def get_flat_factors(prod_expr):
    assert isinstance(prod_expr, pp.Product)
    children = []

    for ch in prod_expr.children:
        if isinstance(ch, pp.Product):
            children.extend(get_flat_factors(ch))
        else:
            children.append(ch)

    return children


def is_derivative_binding(expr):
    return (isinstance(expr, p.OperatorBinding)
            and isinstance(expr.op, p.DerivativeOperator))


def find_inner_deriv_and_coeff(expr):
    if is_derivative_binding(expr):
        return 1, expr
    elif isinstance(expr, pp.Product):
        factors = get_flat_factors(expr)

        derivatives = []
        nonderivatives = []
        for f in factors:
            if is_derivative_binding(f):
                derivatives.append(f)
            else:
                nonderivatives.append(f)

        if len(derivatives) > 1:
            raise ValueError("multiplied second derivatives in '%s'"
                    % expr)

        if not derivatives:
            # We'll only get called if there *is* a second derivative.
            # That we can't find it by picking apart the top-level
            # product is bad news.

            raise ValueError("second derivative inside nonlinearity "
                    "in '%s'" % expr)

        derivative, = derivatives

        return pp.flattened_product(nonderivatives), derivative
    else:
        raise ValueError("unexpected node type '%s' inside "
                "second derivative in '%s'"
                % (type(expr).__name__, expr))


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

    num_equations = len(scalar_unknowns)
    ambient_dim = bvp.ambient_dim

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
                        # diffusion
                        coeff, inner_derivative = \
                                find_inner_deriv_and_coeff(outer_deriv_argument)

                        pot_const, pot_expr = pick_off_constants(
                                inner_derivative.argument)
                        pot_index = tc_storage.register_potential(pot_expr)

                        tc_storage.diffusion[
                                i, pot_index,
                                outer_deriv_axis,
                                inner_derivative.op.ambient_axis] \
                                        += pot_const*coeff

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

    # Python code we generate, we create
    # references to the coefficient arrays in the dictionary
    # that will conveniently have the same name as our pymbolic variables.
    # This makes printing easy and has no major performance penalty.
    defs_list = ["    %s = c[('u', %d)]" % (str(v), i)
                 for (i, v) in enumerate(scalar_unknowns)]

    import string
    defs = string.join(defs_list, '\n')     # noqa

    unk_scalar_fields = [p.Field(psi) for psi in scalar_unknowns]

    def process_scalar_bin(holder, label):
        assign = []
        dassign = []
        deplabels = np.zeros((num_equations, num_equations), 'O')
        deplabels[:] = 'none'
        for (i, x) in enumerate(holder):
            if x:
                xstr = "c[('%s', %d)][:] = %s" % (label, i, x)
                assign.append(xstr)
                for j, psi in enumerate(unk_scalar_fields):
                    dx = differentiate(x, psi)
                    if dx:
                        deplabels[i][j] = classify_dep(dx)
                        dxstr = "c[('d%s', %d, %d)][:] = %s" % (label, i, j, dx)
                        dassign.append(dxstr)
                    else:
                        pass

        return assign, dassign, deplabels

    mass_assigns, dmass_assigns, mass_deps \
        = process_scalar_bin(tc_storage.mass, "m")

    for md in mass_deps.ravel():
        if md == 'constant':
            raise Exception("Constant mass illegal")

    reaction_assigns, dreaction_assigns, reaction_deps \
        = process_scalar_bin(tc_storage.reaction, "r")

    hamiltonian_assigns, dhamiltonian_assigns, hamiltonian_deps \
        = process_scalar_bin(tc_storage.hamiltonian, "h")

    advect_assigns = []
    dadvect_assigns = []

    advect_deps_p = np.zeros((num_equations, num_equations, ambient_dim), 'O')
    advect_deps_p[:] = 'none'

    for i, bi in enumerate(tc_storage.advection):
        for j, bij in enumerate(bi):
            if bij:
                bstr = "c[('f', %d)][..., %d] = %s" % (i, j, bij)
                advect_assigns.append(bstr)
                for k, psi in enumerate(unk_scalar_fields):
                    dbij = differentiate(bij, psi)
                    if dbij:
                        advect_deps_p[i, k, j] = classify_dep(bij)
                        dbstr = "c[('df', %d, %d)][...,%d] = %s" % (i, k, j, dbij)
                        dadvect_assigns.append(dbstr)

    # now "reduce" over the vector component dependences and take the worst.
    dep2int = {'none': 0,
               'constant':  1,
               'linear':    2,
               'nonlinear': 3}
    int2dep = {0: 'none',
               1: 'constant',
               2: 'linear',
               3: 'nonlinear'}

    advect_deps = np.zeros((num_equations, num_equations), "O")
    for i in range(num_equations):
        for j in range(num_equations):
            advect_deps[i, j] = int2dep[
                                    reduce(max,
                                       (dep2int[x]
                                        for x in advect_deps_p[i, j, :]), 0)]

    diff_assigns = []
    ddiff_assigns = []
    diff_deps_p = np.zeros((num_equations,
                            num_equations,
                            num_equations,
                            ambient_dim,
                            ambient_dim), 'O')

    diff_deps_p[:] = 'none'

    for i, ai in enumerate(tc_storage.diffusion):
        for j, aij in enumerate(ai):
            for k, aijk in enumerate(aij):
                for ell, aijkell in enumerate(aijk):
                    if aijkell:
                        astr = "c[('a', %d, %d)[..., %d, %d] = %s" \
                               % (i, j, k, ell, aijkell)
                        diff_assigns.append(astr)
                        for q, psi in enumerate(unk_scalar_fields):
                            da = differentiate(aijkell, psi)
                            diff_deps_p[i, j, q, k, ell] = classify_dep(da)
                            dastr = "c[('da',%d,%d,%d)[...,%d,%d] = %s" \
                                    % (i, j, q, k, ell, da)
                            ddiff_assigns.append(dastr)

    diff_deps = np.zeros((num_equations,
                          num_equations,
                          num_equations), 'O')

    ddp = diff_deps_p.reshape((num_equations,
                               num_equations,
                               num_equations,
                               ambient_dim**2))

    for i in range(num_equations):
        for j in range(num_equations):
            for k in range(num_equations):
                diff_deps[i, j, k] = int2dep[
                                        reduce(max,
                                            (dep2int[x] for x in ddp[i, j, k]), 0)]

    def spacer(x):
        return "    " + x

#    print mass_deps
#    print advect_deps
#    print diff_deps
#    print reaction_deps
#    print hamiltonian_deps

    assigns = reduce(lambda x, y: x+y,
                     [mass_assigns, dmass_assigns,
                      advect_assigns, dadvect_assigns,
                      diff_assigns, ddiff_assigns,
                      reaction_assigns, dreaction_assigns,
                      hamiltonian_assigns, dhamiltonian_assigns])

    for a in assigns:
        print spacer(a)

    # TODO: just need to figure out the potentials since they work

    potential_assign_list = ["c[('phi', %d)][:] = %s" % (i, phi)   # noqa
                             for i, phi in enumerate(tc_storage.potential) if phi]

    # Then we need to figure out the __init__ method.

    # Then we can pretty-print the whole thing.

#  vim: foldmethod=marker
