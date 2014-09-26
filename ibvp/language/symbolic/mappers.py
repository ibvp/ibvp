"""BVP/IBVP description language: language primitives."""

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


from pymbolic.mapper import (
        IdentityMapper as IdentityMapperBase,
        CombineMapper as CombineMapperBase,
        )
from pymbolic.mapper.stringifier import (
        CSESplittingStringifyMapperMixin,
        StringifyMapper as StringifyMapperBase)
from pymbolic.mapper.evaluator import (
        EvaluationMapper as EvaluationMapperBase)
import ibvp.language.symbolic.primitives as p
from pymbolic.mapper.stringifier import PREC_NONE
from pymbolic.geometric_algebra import MultiVector
import numpy as np


# {{{ structural mappers

class OperatorBindingMixin(object):
    def map_operator_binding(self, expr, *args, **kwargs):
        try:
            meth = getattr(self, expr.op.mapper_method+"_binding")
        except AttributeError:
            return super(OperatorBindingMixin, self).map_operator_binding(expr)
        else:
            return meth(expr, *args, **kwargs)


class IdentityMapperOperatorBindingBase(IdentityMapperBase):
    """This class exists so ``super()`` in the :class:`OperatorBindingMixin`
    can find :meth:`map_operator_binding`.
    """

    def map_operator_binding(self, expr, *args, **kwargs):
        return p.OperatorBinding(
                self.rec(expr.op, *args, **kwargs),
                *tuple(self.rec(arg, *args, **kwargs) for arg in expr.arguments))


class IdentityMapper(OperatorBindingMixin, IdentityMapperOperatorBindingBase):
    def map_time_derivative(self, expr):
        return expr

    def map_derivative(self, expr):
        return expr

    def map_parameter(self, expr):
        return expr

    def map_field(self, expr):
        return expr

    map_div = map_derivative
    map_grad = map_derivative
    map_curl = map_derivative


class CombineMapper(CombineMapperBase):
    pass

# }}}


# {{{ stringifiers

class StringifyMapper(StringifyMapperBase):
    def map_operator_binding(self, expr, enclosing_prec):
        return "<%s>(%s)" % (
                self.rec(expr.op, PREC_NONE),
                ", ".join(
                    self.rec(arg, PREC_NONE)
                    for arg in expr.arguments))

    def map_time_derivative(self, expr, enclosing_prec):
        return "d/dt"

    def map_derivative(self, expr, enclosing_prec):
        AXES = "xyz"
        try:
            return "d/d%s" % AXES[expr.ambient_axis]
        except IndexError:
            return "d/dx%d" % expr.ambient_axis

    def map_field(self, expr, enclosing_prec):
        return expr.name

    def map_parameter(self, expr, enclosing_prec):
        return expr.name

    def map_div(self, expr, enclosing_prec):
        return type(expr).__name__[1:].lower()

    map_grad = map_div
    map_curl = map_div

    def map_nabla(self, expr, enclosing_prec):
        return r"\/[%s]" % expr.nabla_id

    def map_nabla_component(self, expr, enclosing_prec):
        return r"d/dx%d[%s]" % (expr.ambient_axis, expr.nabla_id)

    def map_derivative_source(self, expr, enclosing_prec):
        return r"D[%s](%s)" % (expr.nabla_id, self.rec(expr.operand, PREC_NONE))


class PrettyStringifyMapper(
        CSESplittingStringifyMapperMixin,
        StringifyMapper,
        ):
    pass

# }}}


# {{{ evaluation mapper

class EvaluationMapper(EvaluationMapperBase):
    """Unlike :mod:`pymbolic.mapper.evaluation.EvaluationMapper`, this class
    does evaluation mostly to get :class:`pymbolic.geometric_algebra.MultiVector`
    instances to to do their thing, and perhaps to automatically kill terms
    that are multiplied by zero. Otherwise it intends to largely preserve
    the structure of the input expression.
    """

    def map_operator_binding(self, expr, *args, **kwargs):
        return p.OperatorBinding(
                self.rec(expr.op, *args, **kwargs),
                *tuple(self.rec(arg, *args, **kwargs) for arg in expr.arguments))

    def map_time_derivative(self, expr):
        return expr

    def map_derivative(self, expr):
        return expr

    def map_parameter(self, expr):
        return expr

    def map_field(self, expr):
        return expr

    map_div = map_derivative
    map_grad = map_derivative
    map_curl = map_derivative

    def map_common_subexpression(self, expr):
        return p.cse(
                self.rec(expr.child),
                expr.prefix,
                expr.scope)


# }}}


# {{{ scalarizer

class Scalarizer(OperatorBindingMixin, EvaluationMapper):
    def __init__(self, ambient_dim):
        # FIXME: Might be better to make 'ambient_dim' a per-domain
        # thing.
        EvaluationMapper.__init__(self)
        self.ambient_dim = ambient_dim

    def map_nabla(self, expr):
        from pytools.obj_array import make_obj_array
        return MultiVector(make_obj_array(
            [p.NablaComponent(axis, expr.nabla_id)
                for axis in xrange(self.ambient_dim)]))

    def map_vector_variable(self, expr):
        from pymbolic import make_sym_vector
        num_components = expr.num_components

        if num_components is None:
            num_components = self.ambient_dim

        # return MultiVector(make_sym_vector(expr.name, num_components))
        return make_sym_vector(expr.name, self.ambient_axis)

    def map_div_binding(self, expr):
        rec_arg = self.rec(expr.argument)
        assert isinstance(rec_arg, np.ndarray)
        return sum(
                p.DerivativeOperator(i)(expr_i)
                for i, expr_i in enumerate(rec_arg))
        # d = p.Derivative()
        # arg = self.rec(expr.argument)
        # z = self.rec(d.nabla).scalar_product(d(arg))
        # print "DIV", self.rec(d.nabla)
        # print "DIV", d(arg)
        # return z

    def map_grad_binding(self, expr):
        from pytools.obj_array import make_obj_array
        rec_arg = self.rec(expr.argument)
        return make_obj_array([
            p.DerivativeOperator(i)(rec_arg)
            for i in range(self.ambient_dim)])

        # d = p.Derivative()
        # z = self.rec(d.nabla)*d(expr.argument)
        # print "ZZZ", z
        # return z

    def map_curl_binding(self, expr):
        raise NotImplementedError()

# }}}

# vim: foldmethod=marker
