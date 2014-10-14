def main():
    import ibvp.sym as sym
    import numpy as np

    ambient_dim = 2

    p = 1
    q = 2
    r = 2

    u = sym.Field("u")

    B = np.array([1.0, 2.0])

    Adiff = 0.001
    C = 0.0001

    eqns = sym.join(
            sym.d_dt(u**p)
            + sym.div(B * u**q)
            - sym.div(Adiff * sym.grad(u**r))
            + C * u
            )

    print sym.pretty(eqns)

    from ibvp.language import IBVP
    from ibvp.target.proteus import generate_proteus_problem_file

    generate_proteus_problem_file(
            IBVP(
                ambient_dim=ambient_dim,
                pde_system=eqns,
                unknowns=[u.name],
                ),
            "Burgers")


if __name__ == "__main__":
    main()
