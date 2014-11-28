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


import numpy as np


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


def join_fields(*args):
    from pytools.obj_array import make_obj_array, log_shape
    from pymbolic.geometric_algebra import MultiVector, bit_count

    res_list = []
    for arg in args:
        if isinstance(arg, list):
            res_list.extend(arg)

        elif isinstance(arg, MultiVector):
            for grade in arg.all_grades():
                for bits in range(2**arg.space.dimensions):
                    if bit_count(bits) == grade:
                        res_list.append(arg.data.get(bits, 0))

        elif isinstance(arg, np.ndarray):
            if log_shape(arg) == ():
                res_list.append(arg)
            else:
                res_list.extend(arg.flat)
        else:
            res_list.append(arg)

    return make_obj_array(res_list)
