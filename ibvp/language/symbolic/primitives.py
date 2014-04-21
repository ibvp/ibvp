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


class Expression(p.Expression):
    def stringifier(self):
        from ibvp.language.symbolic.mappers import StringifyMapper
        return StringifyMapper


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

# vim: foldmethod=marker
