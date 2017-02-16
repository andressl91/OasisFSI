from fenics import Identity, grad, tr

# First Piola Kirchoff stress tensor
def Piola1(d_, w_, lambda_, mu_s, E_func=None):
    I = Identity(2)
    if callable(E_func):
        E = E_func(d_, w_)
    else:
        F = I + grad(d_["n"])
        E = 0.5*((F.T*F) - I)

    return F*(lambda_*tr(E)*I + 2*mu_s*E)

#Second Piola Kirchhoff Stress tensor
def Piola2(d_, w_, k, lambda_, mu_s, E_func=None):
    I = Identity(2)
    if callable(E_func):
        E = E_func(d_, w_, k)
    else:
        F = I + grad(d_["n"])
        E = 0.5*((F.T*F) - I)

    return lambda_*tr(E)*I + 2*mu_s*E


def reference(d_, w_, k):
    E = 0.5*(grad(d_["n"]).T + grad(d_["n-1"]).T \
             + grad(d_["n"]) + grad(d_["n-1"])) \
        + 0.5*(grad(d_["n"]).T * grad(d_["n"])) \
        + 0.5*(grad(d_["n-1"]).T * grad(d_["n-1"]))

    return 0.5*E


def naive_linearization(d_, w_, k):
    E = grad(d_["n"]) + grad(d_["n"]).T \
        + grad(d_["n-1"]).T*grad(d_["n"])

    return 0.5*E


def explicit(d_, w_, k):
    E = grad(d_["n-1"]) + grad(d_["n-1"]).T \
        + grad(d_["n-1"]).T*grad(d_["n-1"])

    return 0.5*E


def naive_ab(d_, w_, k):
    E = 0.5*(grad(d_["n"]).T + grad(d_["n-1"]).T \
            + grad(d_["n"]) + grad(d_["n-1"])) \
        + 3./2 * (grad(d_["n-1"]).T*grad(d_["n-1"])) \
        - 1./2 * (grad(d_["n-2"]).T*grad(d_["n-2"]))

    return 0.5*E


def ab_before_cn(d_, w_, k):
    E = 0.5*(grad(d_["n"]).T + grad(d_["n-1"]).T \
        + grad(d_["n"]) + grad(d_["n-1"])) \
        + (3/2.*grad(d_["n-1"]).T - 0.5*grad(d_["n-2"]).T)\
        * 0.5*(grad(d_["n"]) + grad(d_["n-1"]))

    return 0.5*E


def ab_before_cn_higher_order(d_, w_, k):
    E = 0.5*(grad(d_["n"]).T + grad(d_["n-1"]).T \
        + grad(d_["n"]) + grad(d_["n-1"])) \
        + (23/12.*grad(d_["n-1"]).T - 4./3*grad(d_["n-2"]).T + 5/12.*grad(d_["n-3"]).T) \
        * 0.5*(grad(d_["n"]) + grad(d_["n-1"]))

    return 0.5*E


def cn_before_ab(d_, w_, k):
    E = 0.5*(grad(d_["n"]).T + grad(d_["n-1"]).T + grad(d_["n"]) + grad(d_["n-1"]))  \
        + 0.5*((grad(d_["n-1"] + k*(3./2*w_["n-1"] - 1./2*w_["n-2"])).T * grad(d_["n"])) \
                + (grad(d_["n-1"]).T*grad(d_["n-1"])))

    return 0.5*E


def cn_before_ab_higher_order(d_, w_, k):
    E = 0.5*(grad(d_["n"]).T + grad(d_["n-1"]).T + grad(d_["n"]) + grad(d_["n-1"]))  \
        + 0.5*((grad(d_["n-1"] + k*(23./12*w_["n-1"] - 4./3*w_["n-2"] + 5/12.*w_["n-3"])).T \
                * grad(d_["n"])) + (grad(d_["n-1"]).T*grad(d_["n-1"])))

    return 0.5*E
