def main():
    import ibvp.sym as sym

    ambient_dim = 2

    u = sym.Field("u")

    heat_eqn = (
            sym.d_dt(u)
            - sym.div(sym.grad(u)))

    print heat_eqn

    # Now perform a (nonsenical) transformation that multiplies all (spatial)
    # derivatives by two.

    from ibvp.language.symbolic.mappers import (
            IdentityMapper, Scalarizer)

    scalarized_heat_eqn = Scalarizer(ambient_dim)(heat_eqn)

    print sym.pretty(scalarized_heat_eqn)

    return
    class SpatialDerivativeDoubler(IdentityMapper):
        def map_derivative_binding(self, expr):
            return 2*expr.op(self.rec(expr.argument))

        map_div_binding = map_derivative_binding
        map_grad_binding = map_derivative_binding
        map_curl_binding = map_derivative_binding

    print sym.pretty(
            SpatialDerivativeDoubler()(heat_eqn))


if __name__ == "__main__":
    main()
