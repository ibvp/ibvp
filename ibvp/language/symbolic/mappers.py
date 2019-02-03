"""BVP/IBVP description language: language primitives."""

from __future__ import division, absolute_import

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


from six.moves import range

from pymbolic.mapper.dependency import (
        DependencyMapper as DependencyMapperBase)

from pymbolic.geometric_algebra.mapper import (
        IdentityMapper as IdentityMapperBase,
        CombineMapper as CombineMapperBase,
        Collector as CollectorBase,
        WalkMapper as WalkMapperBase,
        EvaluationMapper as EvaluationMapperBase,
        StringifyMapper as StringifyMapperBase,
        Dimensionalizer,
        DerivativeBinder as DerivativeBinderBase,

        DerivativeSourceAndNablaComponentCollector
        as DerivativeSourceAndNablaComponentCollectorBase,
        NablaComponentToUnitVector
        as NablaComponentToUnitVectorBase,
        DerivativeSourceFinder
        as DerivativeSourceFinderBase,
        ConstantFoldingMapper
        as ConstantFoldingMapperBase,

        )
from pymbolic.mapper.differentiator import (
        DifferentiationMapper as DifferentiationMapperBase,)
from pymbolic.mapper.stringifier import (
        CSESplittingStringifyMapperMixin,)
from pymbolic.mapper.distributor import (
        DistributeMapper as DistributeMapperBase)
import ibvp.language.symbolic.primitives as p
import pymbolic.primitives as pp
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
        new_op = self.rec(expr.op, *args, **kwargs)
        return new_op(
                *tuple(self.rec(arg, *args, **kwargs)
                    for arg in expr.arguments))


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
    def map_operator_binding(self, expr):
        return self.combine([
            self.rec(expr.op), self.rec(expr.argument)])


class Collector(CollectorBase, CombineMapper):
    # All leaves
    def map_field(self, expr):
        return set()

    map_time = map_field
    map_vector_field = map_field
    map_time_derivative = map_field
    map_parameter = map_field
    map_derivative = map_field


class DependencyMapper(Collector, DependencyMapperBase):
    pass


class WalkMapper(WalkMapperBase):
    def map_operator_binding(self, expr, *args):
        if not self.visit(expr, *args):
            return

        self.rec(expr.op, *args)
        self.rec(expr.argument, *args)

    def map_time_derivative(self, expr, *args):
        # A leaf.
        if not self.visit(expr, *args):
            return

    map_derivative = map_time_derivative

    map_field = map_time_derivative
    map_vector_field = map_time_derivative
    map_parameter = map_time_derivative

    map_div = map_time_derivative
    map_grad = map_time_derivative
    map_curl = map_time_derivative

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
        AXES = "xyz"  # noqa: N806
        try:
            return "d/d%s" % AXES[expr.ambient_axis]
        except IndexError:
            return "d/dx%d" % expr.ambient_axis

    def map_field(self, expr, enclosing_prec):
        return expr.name

    map_parameter = map_field
    map_vector_field = map_field
    map_multivector_field = map_field

    def map_div(self, expr, enclosing_prec):
        return type(expr).__name__[1:].lower()

    map_grad = map_div
    map_curl = map_div


class PrettyStringifyMapper(
        CSESplittingStringifyMapperMixin,
        StringifyMapper,
        ):
    pass

# }}}


# {{{ distribute mapper

class DistributeMapper(DistributeMapperBase):
    def __init__(self):
        super(DistributeMapper, self).__init__(
                collector=lambda expr: expr,
                const_folder=ConstantFoldingMapper())

    def map_operator_binding(self, expr):
        rec_arg = self.rec(expr.argument)

        if isinstance(expr.op, p.LinearOperator):
            if isinstance(rec_arg, pp.Sum):
                return pp.Sum(
                        tuple(
                            expr.op(term)
                            for term in rec_arg.children))
            else:
                return expr.op(rec_arg)
        else:
            return expr.op(rec_arg)

    def map_field(self, expr):
        return expr

    map_vector_field = map_field
    map_parameter = map_field

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
        new_op = self.rec(expr.op, *args, **kwargs)
        return new_op(
                *tuple(self.rec(arg, *args, **kwargs)
                    for arg in expr.arguments))

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
        return pp.CommonSubexpression(
                self.rec(expr.child),
                expr.prefix,
                expr.scope)


# }}}


# {{{ derivative binder

class DerivativeSourceAndNablaComponentCollector(
        DerivativeSourceAndNablaComponentCollectorBase,
        Collector):
    pass


class NablaComponentToUnitVector(
        NablaComponentToUnitVectorBase,
        EvaluationMapper):
    pass


class DerivativeSourceFinder(
        DerivativeSourceFinderBase,
        EvaluationMapper):
    pass


class DerivativeBinder(DerivativeBinderBase, IdentityMapper):
    derivative_source_and_nabla_component_collector = \
            DerivativeSourceAndNablaComponentCollector
    nabla_component_to_unit_vector = NablaComponentToUnitVector
    derivative_source_finder = DerivativeSourceFinder

    def take_derivative(self, ambient_axis, expr):
        return p.DerivativeOperator(ambient_axis)(expr)

# }}}


# {{{ scalarizer

class Scalarizer(OperatorBindingMixin, Dimensionalizer, EvaluationMapper):
    def __init__(self, ambient_dim):
        # FIXME: Might be better to make 'ambient_dim' a per-domain
        # thing.
        EvaluationMapper.__init__(self)
        self._ambient_dim = ambient_dim

    @property
    def ambient_dim(self):
        return self._ambient_dim

    def map_curl_binding(self, expr):
        raise NotImplementedError()

    def map_numpy_array(self, expr):
        if len(expr.shape) != 1:
            raise ValueError("only 1D numpy arrays are supported")

        from pytools.obj_array import join_fields
        return join_fields(*[
            self.rec(expr[i])
            for i in range(len(expr))])

    def map_subscript(self, expr, *args):
        return p.Field("%s_%s" % (expr.aggregate, expr.index))

    # {{{ conventional vector calculus

    def map_vector_field(self, expr):
        return self.rec(p.make_field_vector(expr.name, self.ambient_dim))

    def map_div_binding(self, expr):
        rec_arg = self.rec(expr.argument)
        assert isinstance(rec_arg, np.ndarray)
        return sum(
                p.DerivativeOperator(i)(expr_i)
                for i, expr_i in enumerate(rec_arg))

    def map_grad_binding(self, expr):
        from pytools.obj_array import make_obj_array
        rec_arg = self.rec(expr.argument)
        return make_obj_array([
            p.DerivativeOperator(i)(rec_arg)
            for i in range(self.ambient_dim)])

    # }}}

    # {{{ geometric calculus

    def map_multivector_field(self, expr):
        from pymbolic.primitives import make_sym_vector
        return MultiVector(
                make_sym_vector(
                    expr.name, self.ambient_dim,
                    var_factory=p.Field))

    # }}}

    def __call__(self, expr):
        result = super(Scalarizer, self).__call__(expr)
        return DerivativeBinder()(result)

# }}}


# {{{ differentiation mapper

class DifferentiationMapper(DifferentiationMapperBase):
    map_field = DifferentiationMapperBase.map_variable
    map_parameter = DifferentiationMapperBase.map_variable


def differentiate(expr, var):
    return DifferentiationMapper(var)(expr)

# }}}


# {{{ constant folder

class ConstantFoldingMapper(IdentityMapper, ConstantFoldingMapperBase):
    def is_constant(self, expr):
        return not bool(DependencyMapper()(expr))

# }}}


# vim: foldmethod=marker
