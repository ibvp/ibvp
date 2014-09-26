"""Strong-form BVP/IVP description language."""

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

from pytools import memoize_method


# {{{ problems

class BVP(object):
    """
    .. attribute :: ambient_dim

    .. attribute :: pde_system

        A :class:`numpy.ndarray` of :class:`pymbolic.primitives.Expression`.

    .. attribute:: unknowns

        A list of identifiers for which a solution is desired.
    """

    def __init__(self, ambient_dim, pde_system, unknowns):
        self.ambient_dim = ambient_dim
        self.pde_system = pde_system
        self.unknowns = unknowns


class IBVP(BVP):
    pass

# }}}

# vim: foldmethod=marker
