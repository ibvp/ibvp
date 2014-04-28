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
import ibvp.language.symbolic.primitives as p


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


class CombineMapper(CombineMapperBase):
    pass


class StringifyMapper(StringifyMapperBase):
    def map_operator_binding(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_NONE
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


class PrettyStringifyMapper(
        CSESplittingStringifyMapperMixin,
        StringifyMapper,
        ):
    pass

# vim: foldmethod=marker
