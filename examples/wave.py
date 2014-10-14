def main():
    import ibvp.sym as sym

    # Currently broken since the parameters are not properly
    # stringified.
    1/0

    dim = 3
    u = sym.Field("u")
    f = sym.Field("f")
    c = sym.Parameter("c")
    v = sym.VectorField("v")

    eqns = sym.join(
            sym.d_dt(u)
            + c * sym.div(v) - f,

            sym.d_dt(v)
            + c * sym.grad(u)
            )

    print sym.pretty(eqns)

    from ibvp.language import IBVP
    from ibvp.target.proteus import generate_proteus_problem_file
    generate_proteus_problem_file(
            IBVP(
                ambient_dim=dim,
                pde_system=eqns,
                unknowns=[u.name, v.name],
                ),
            "Wave")


if __name__ == "__main__":
    main()
