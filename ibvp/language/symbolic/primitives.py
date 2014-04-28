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


import pymbolic.primitives as p
import numpy as np


class Expression(p.Expression):
    def stringifier(self):
        from ibvp.language.symbolic.mappers import StringifyMapper
        return StringifyMapper


# {{{ operators and binding

class Operator(Expression):
    def __call__(self, expr):
        from pytools.obj_array import with_object_array_or_scalar

        def bind_one(subexpr):
            if p.is_zero(subexpr):
                return subexpr
            else:
                return OperatorBinding(self, subexpr)

        return with_object_array_or_scalar(bind_one, expr)


class OperatorBinding(Expression):
    def __init__(self, op, *arguments):
        self.op = op
        self.arguments = arguments

    @property
    def argument(self):
        assert len(self.arguments) == 1
        return self.arguments[0]

    mapper_method = intern("map_operator_binding")

    def __getinitargs__(self):
        return self.op, self.arguments


class Parameter(p.Variable):
    mapper_method = intern("map_parameter")


class Field(p.Variable):
    mapper_method = intern("map_field")


def make_field_vector(name, components):
    """Return an object array of *components* subscripted
    :class:`Variable` instances.

    :param components: The number of components in the vector.
    """
    if isinstance(components, int):
        components = range(components)

    from pytools.obj_array import join_fields
    vfld = Field(name)
    return join_fields(*[vfld[i] for i in components])

# }}}


# {{{ functions

class Function(p.Variable, Expression):
    def __call__(self, operand, *args, **kwargs):
        # If the call is handed an object array full of operands,
        # return an object array of the function applied to each of the
        # operands.

        from pytools.obj_array import is_obj_array, with_object_array_or_scalar
        if is_obj_array(operand):
            def make_op(operand_i):
                return self(operand_i, *args, **kwargs)

            return with_object_array_or_scalar(make_op, operand)
        else:
            return p.Variable.__call__(self, operand)

real = Function("real")
imag = Function("imag")
conj = Function("conj")
sqrt = Function("sqrt")
exp = Function("exp")
sin = Function("sin")
cos = Function("cos")
tan = Function("tan")
sinh = Function("sinh")
cosh = Function("cosh")
arcsin = Function("arcsin")
arccos = Function("arccos")
arctan = Function("arctan")
arctan2 = Function("arctan2")
abs = Function("abs")

# }}}


# {{{ discretization properties

class GeometryProperty(Expression):
    """A quantity that depends exclusively on the geometry (and has no
    further arguments.
    """

    def __init__(self, where=None):
        """
        :arg where: a symbolic name of a :class:`GeometryComponent`
        """

        self.where = where

    def __getinitargs__(self):
        return (self.where,)


class CoordinateComponent(GeometryProperty):
    def __init__(self, ambient_axis, where=None):
        """
        :arg where: a symbolic name of a :class:`GeometryComponent`
        """
        self.ambient_axis = ambient_axis
        GeometryProperty.__init__(self, where)

    mapper_method = intern("map_coordinate_component")


class CoordinateVector(GeometryProperty):
    def __init__(self, where=None):
        GeometryProperty.__init__(self, where)

    mapper_method = intern("map_nodes")


class BoundaryNormalComponent(GeometryProperty):
    def __init__(self, where=None):
        GeometryProperty.__init__(self, where)

    mapper_method = intern("map_boundary_normal_component")


class BoundaryNormalVector(GeometryProperty):
    def __init__(self, where=None):
        GeometryProperty.__init__(self, where)

    mapper_method = intern("map_boundary_normal")

# }}}


# {{{ time

class Time(Expression):
    mapper_method = "map_time"


class TimeDerivativeOperator(Operator):
    mapper_method = "map_time_derivative"

d_dt = TimeDerivativeOperator()

# }}}


# {{{ spatial calculus

class DerivativeOperator(Operator):
    def __init__(self, ambient_axis):
        self.ambient_axis = ambient_axis

    mapper_method = "map_derivative"


d_dx = DerivativeOperator(0)
d_dy = DerivativeOperator(1)
d_dz = DerivativeOperator(2)


def div(expr):
    return sum(DerivativeOperator(i)(e_i) for i, e_i in enumerate(expr))


def grad(dim, expr):
    from pytools.obj_array import log_shape, with_object_array_or_scalar
    base_shape = log_shape(expr)

    result = np.zeros((dim,)+base_shape, dtype=np.object)

    for i in range(dim):
        result[i] = with_object_array_or_scalar(
                DerivativeOperator(i), expr)

    return result


def curl(expr, cross_product=None):
    if cross_product is None:
        from ibvp.language.symbolic.util import SubsettableCrossProduct
        cross_product = SubsettableCrossProduct()

    def three_mult(sign, op, field):
        return sign*op(field)

    return cross_product(
            [DerivativeOperator(i) for i in range(3)],
            expr)

# }}}


# vim: foldmethod=marker
