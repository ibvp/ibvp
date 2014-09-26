def main():
    import ibvp.sym as sym

    ambient_dim = 2

    u = sym.Field("u")
    v = sym.Field("v")

    eqns = sym.join(
            sym.d_dt(u) - sym.div(sym.grad(u-v)),
            sym.d_dt(v) - sym.div(sym.grad(u+v)),
            )

    print sym.pretty(eqns)

    from ibvp.language import IBVP
    from ibvp.target.proteus import generate_proteus_problem_file

    generate_proteus_problem_file(
            IBVP(
                ambient_dim=ambient_dim,
                pde_system=eqns,
                unknowns=[u.name, v.name],
                ))


if __name__ == "__main__":
    main()
