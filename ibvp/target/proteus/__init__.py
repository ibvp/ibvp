"""Proteus target for IBVP translation."""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from six.moves import map
from six.moves import range
from six.moves import zip
from functools import reduce

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
        DistributeMapper, CombineMapper,
        differentiate)
from ibvp.language.symbolic.util import pretty
import ibvp.language.symbolic.primitives as p
import pymbolic.primitives as pp
from pymbolic.mapper.dependency import DependencyMapper as DepMapBase


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

    def __init__(self, scalarized_bvp, adim, scalar_unknowns):
        self.scalarized_bvp = scalarized_bvp

        self.scalar_unknowns = scalar_unknowns

        neq = len(scalarized_bvp.pde_system)

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

# }}}


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


# }}}


def generate_proteus_problem_file(bvp, clsnm):
    from ibvp.language import scalarize
    scalarized_system = scalarize(bvp)

    import ibvp.sym as sym
    print(sym.pretty(scalarized_system.pde_system))

    distr_system = DistributeMapper()(scalarized_system.pde_system)

    scalar_unknowns = scalarized_system.unknowns

    num_equations = len(scalar_unknowns)
    ambient_dim = bvp.ambient_dim

    if len(set(scalar_unknowns)) != len(scalar_unknowns):
        raise ValueError("names of unknowns not unique "
                "after scalarization")

#    import ibvp.sym as sym
#    print sym.pretty(distr_system)

    tc_storage = TransportCoefficientStorage(scalarized_system,
                                             bvp.ambient_dim,
                                             scalar_unknowns)

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

    defs = '\n'.join(defs_list)  # noqa

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
                        if deplabels[i][j] == 'nonlinear' \
                           or deplabels[i][j] == 'linear':
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
                        astr = "c[('a', %d, %d)][..., %d, %d] = %s" \
                               % (i, j, k, ell, aijkell)
                        diff_assigns.append(astr)
                        for q, psi in enumerate(unk_scalar_fields):
                            da = differentiate(aijkell, psi)
                            if da:
                                diff_deps_p[i, j, q, k, ell] = classify_dep(da)
                                dastr = "c[('da',%d,%d,%d)][...,%d,%d] = %s" \
                                        % (i, j, q, k, ell, da)
                                ddiff_assigns.append(dastr)
                            else:
                                diff_deps_p[i, j, q, k, ell] = 'constant'

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

    # potential is a bit different from other scalars.
    potential_assigns = []
    dpotential_assigns = []

    phi_deps = np.zeros((num_equations, num_equations), 'O')
    for i, phi in enumerate(tc_storage.potential):
        for j, u in enumerate(unk_scalar_fields):
            if phi == u:
                phi_deps[i, j] = 'u'
            else:
                phi_str = "c[('phi', %d)] = %s" % (i, phi)
                potential_assigns.extend(phi_str)
                D = differentiate(phi, u)
                if D:
                    phi_deps[i, j] = 'nonlinear'
                    dphi_str = "c[('dphi', %d, %d)] = %s" % (i, j, D)
                    dpotential_assigns.extend(dphi_str)

    def spacer(x):
        return "        " + x

    assigns = "\n".join(
                map(spacer,
                    reduce(lambda x, y: x+y,
                           [mass_assigns, dmass_assigns,
                            advect_assigns, dadvect_assigns,
                            diff_assigns, ddiff_assigns,
                            reaction_assigns, dreaction_assigns,
                            hamiltonian_assigns, dhamiltonian_assigns])))

    # we dict-ify the dependencies so we can repr them.
    def dictify(arr):
        if len(arr.shape) == 1:
            return dict((i, a) for (i, a) in enumerate(arr) if a and a != 'none')
        else:
            result = {}
            for i, a in enumerate(arr):
                da = dictify(a)
                if len(da) > 0:
                    result[i] = da
            return result

    nms = ["mass", "advection", "diffusion", "potential", "reaction", "hamiltonian"]
    deps = [mass_deps, advect_deps, diff_deps, phi_deps,
            reaction_deps, hamiltonian_deps]

    dep_stmnts = []
    for (nm, d) in zip(nms, deps):
        ddict = dictify(d)
        dep_stmnts.append("        %s = %s" % (nm, repr(ddict)))

    dep_st = "\n".join(dep_stmnts)

    # This is for creating, e.g. u = c[('u',0)] before we make assignments
    # in evaluate so that we have references into the c dictionary for our
    # data.  This makes the pretty-printed code more readable.
    ref_list = []
    for i, phi in enumerate(scalar_unknowns):
        ref_list.append("%s = c[('u',%d)]" % (phi, i))

    refs = "\n".join((spacer(x) for x in ref_list))

    tc_class_str = """
from proteus.TransportCoefficients import TC_base

class %s(TC_base):
    def __init__(self):
%s
        variableNames=%s
        TC_base.__init__(self,
                         nc=%d,
                         mass=mass,
                         advection=advection,
                         diffusion=diffusion,
                         potential=potential,
                         reaction=reaction,
                         hamiltonian=hamiltonian,
                         variableNames=variableNames)

    def evaluate(self, t, c):
%s
%s
""" % (clsnm, dep_st, repr(scalar_unknowns), num_equations, refs, assigns)

    print(tc_class_str)

#  vim: foldmethod=marker
