import numpy as np


def main():
    import ibvp.sym as sym

    ambient_dim = 2

    u = sym.Field("u")
    v = sym.Field("v")

    eqns = np.array([
            sym.d_dt(u) - sym.div(sym.grad(u-v)),
            sym.d_dt(v) - sym.div(sym.grad(u+v)),
            ])

    print sym.pretty(eqns)

    # Now perform a (nonsenical) transformation that multiplies all (spatial)
    # derivatives by two.

    from ibvp.language.symbolic.mappers import (
            Scalarizer)

    scalarized_heat_eqn = Scalarizer(ambient_dim)(eqns)

    print sym.pretty(scalarized_heat_eqn)

    from ibvp.language import IBVP
    from ibvp.target.proteus import generate_proteus_problem_file

    generate_proteus_problem_file(
            IBVP(
                ambient_dim=ambient_dim,
                pde_system=np.array([
                    scalarized_heat_eqn
                    ]),
                unknowns=[u.name, v.name],
                ))


if __name__ == "__main__":
    main()
