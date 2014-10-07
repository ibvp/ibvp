def main():
    import ibvp.sym as sym

    ambient_dim = 2

    u = sym.Field("u")

    eqns = sym.join(
            sym.d_dt(u)
            - sym.div(sym.grad(u))
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
            "Heat")


if __name__ == "__main__":
    main()
