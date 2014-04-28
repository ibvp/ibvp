from __future__ import division

__copyright__ = "Copyright (C) 2010-2013 Andreas Kloeckner"

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


def pretty(expr):
    from ibvp.language.symbolic.mappers import PrettyStringifyMapper
    stringify_mapper = PrettyStringifyMapper()
    from pymbolic.mapper.stringifier import PREC_NONE
    result = stringify_mapper(expr, PREC_NONE)

    splitter = "="*75 + "\n"

    cse_strs = stringify_mapper.get_cse_strings()
    if cse_strs:
        result = "\n".join(cse_strs)+"\n"+splitter+result

    return result


class SubsettableCrossProduct:
    """A cross product that can operate on an arbitrary subsets of its
    two operands and return an arbitrary subset of its result.
    """

    full_subset = (True, True, True)

    def __init__(self, op1_subset=full_subset, op2_subset=full_subset,
            result_subset=full_subset):
        """Construct a subset-able cross product.

        :param op1_subset: The subset of indices of operand 1 to be taken
            into account.
            Given as a 3-sequence of bools.
        :param op2_subset: The subset of indices of operand 2 to be taken
            into account.
            Given as a 3-sequence of bools.
        :param result_subset: The subset of indices of the result that are
            calculated.
            Given as a 3-sequence of bools.
        """
        def subset_indices(subset):
            return [i for i, use_component in enumerate(subset)
                    if use_component]

        self.op1_subset = op1_subset
        self.op2_subset = op2_subset
        self.result_subset = result_subset

        import pymbolic
        op1 = pymbolic.var("x")
        op2 = pymbolic.var("y")

        from pytools import levi_civita

        self.functions = []
        self.component_lcjk = []
        for i, use_component in enumerate(result_subset):
            if use_component:
                this_expr = 0
                this_component = []
                for j, j_real in enumerate(subset_indices(op1_subset)):
                    for k, k_real in enumerate(subset_indices(op2_subset)):
                        lc = levi_civita((i, j_real, k_real))
                        if lc != 0:
                            this_expr += lc*op1[j]*op2[k]
                            this_component.append((lc, j, k))
                self.functions.append(pymbolic.compile(this_expr,
                    variables=[op1, op2]))
                self.component_lcjk.append(this_component)

    def __call__(self, x, y, three_mult=None):
        """Compute the subsetted cross product on the indexables *x* and *y*.

        :param three_mult: a function of three arguments *sign, xj, yk*
          used in place of the product *sign*xj*yk*. Defaults to just this
          product if not given.
        """
        from pytools.obj_array import join_fields
        if three_mult is None:
            return join_fields(*[f(x, y) for f in self.functions])
        else:
            return join_fields(
                    *[sum(three_mult(lc, x[j], y[k]) for lc, j, k in lcjk)
                    for lcjk in self.component_lcjk])
