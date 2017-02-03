def problem_mix(T, dt, E, coupling, VV, **Sold_namespace):
    # Function space
    V = VectorFunctionSpace(mesh, "CG", 2)
    VV=V*V

    # Temporal parameters
    t = 0
    k = Constant(dt)

    # Split problem to two 1.order differential equations
    psi, phi = TestFunctions(VV)

    # BCs
    bc1 = DirichletBC(VV.sub(0), ((0, 0)), boundaries, 1)
    bc2 = DirichletBC(VV.sub(1), ((0, 0)), boundaries, 1)
    bcs = [bc1, bc2]

    # Functions, wd is for holding the solution
    d_ = {}; w_ = {}; wd_ = {}
    for time in ["n", "n-1", "n-2", "n-3"]:
        if time == "n" and E not in [None, reference]:
            tmp_wd = Function(VV)
            wd_[time] = tmp_wd
            wd = TrialFunction(VV)
            w, d = split(wd)
        else:
            wd = Function(VV)
            wd_[time] = wd
            w, d = split(wd)

        d_[time] = d
        w_[time] = w

    # Time derivative
    if coupling == "center":
        G = rho_s/(2*k) * inner(w_["n"] - w_["n-2"], psi)*dx
    else:
        G = rho_s/k * inner(w_["n"] - w_["n-1"], psi)*dx

    # Stress tensor
    G += inner(Piola2(d_, w_, k, E_func=E), grad(psi))*dx

    # Gravity
    G -= inner(g, psi)*dx

    # d-w coupling
    if coupling == "CN":
        G += inner(d_["n"] - d_["n-1"] - k*0.5*(w_["n"] + w_["n-1"]), phi)*dx
    elif coupling == "imp":
        G += inner(d_["n"] - d_["n-1"] - k*w_["n"], phi)*dx
    elif coupling == "exp":
        G += inner(d_["n"] - d_["n-1"] - k*w_["n-1"], phi)*dx
    elif coupling == "center":
        G += innter(d_["n"] - d_["n-2"] - 2*k*w["n-1"], phi)*dx
    else:
        print "The coupling %s is not implemented, 'CN', 'imp', and 'exp' are the only valid choices."
        sys.exit(0)

    # Solve
    if E in [None, reference]:
        displacement_x, displacement_y, time = solver_nonlinear(G, d_, w_, wd_, bcs, T, dt)
    else:
        displacement_x, displacement_y, time = solver_linear(G, d_, w_, wd_, bcs, T, dt)

    return displacement_x, displacement_y, time
