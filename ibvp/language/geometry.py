"""Strong-form BVP/IVP description language: geometry."""

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


# {{{ geometry component

class GeometryComponent(object):
    """
    One component of a :class:`Geometry`.

    .. attribute:: dim

        Dimensionality of the "bulk" of the mesh. See
        :attr:`Geometry.ambient_dim` for the ambient dimension.

    .. attribute:: boundary_names

        iterable with membership test of symbolic boundary
        names.
    """


class GeometryComponentFromDiscretization(GeometryComponent):
    def __init__(self, file_name, file_format=None):
        raise NotImplementedError


class SignedDistanceGeometryComponent(GeometryComponent):
    def __init__(self):
        raise NotImplementedError

# }}}


class Geometry(object):
    """
    An abstract geometry description. Note that this object is not expected to
    capture discretization information of any sort. It can be transformed
    into a :class:`ibvp.discretization.Discretization` using the functions
    in :mod:`ibvp.discretization`. If a geometry is to be given by an already
    existing discretizaiton, see :class:`GeometryFromDiscretization`.

    .. attribute:: ambient_dim

    .. attribute:: components

        Mapping from symbolic name to a :class:`GeometryComponent`

    .. attribute:: connectivity

        Mapping from tuples ``(component, boundary)`` to tuples of ``(component,
        boundary)`` pairs. In each of these, *component* is a symbolic name of
        a :class:`GeometryComponent` and *boundary* is a symbolic name of a
        boundary (see :attr:`GeometryComponent.boundary_names`).

        Note that this allows one boundary of one component to connect up to
        multiple other components' boundaries.

        Connectivity in both directions must be explicitly represented.
    """

# vim: foldmethod=marker
