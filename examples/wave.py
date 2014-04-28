def main():
    import ibvp.sym as sym

    dim = 3
    u = sym.Field("u")
    f = sym.Field("f")
    c = sym.Parameter("c")
    v = sym.make_field_vector("v", dim)

    wave_eqn = sym.join(
            sym.d_dt(u)
            + c * sym.div(v) - f,

            sym.d_dt(v)
            + c * sym.grad(3, u)
            )

    # Now perform a (nonsenical) transformation that multiplies all (spatial)
    # derivatives by two.

    from ibvp.language.symbolic.mappers import IdentityMapper

    class SpatialDerivativeDoubler(IdentityMapper):
        def map_derivative_binding(self, expr):
            return 2*expr.op(expr.argument)

    print sym.pretty(
            SpatialDerivativeDoubler()(wave_eqn))


if __name__ == "__main__":
    main()
