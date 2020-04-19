import casadi as cs

def rk4(f, x0, par, Delta=1, M=1):
    """
    Does M RK4 timesteps of function f with variables x0 and parameters par.

    The first argument of f must be var, followed by any number of parameters
    given in a list in order.

    Note that var and the output of f must add like numpy arrays.

    Function obtained from MPCTools: Nonlinear Model Predictive Control Tools for CasADi (Python Interface)
    https://bitbucket.org/rawlings-group/mpc-tools-casadi/src/9c946cb754782c2e5f7f394adf8bda51e0f603eb/mpctools/util.py#lines-37
    """
    h = Delta / M
    x = x0
    j = 0
    while j < M:  # For some reason, a for loop creates problems here.
        k1 = f(x, *par)
        k2 = f(x + k1 * h / 2, *par)
        k3 = f(x + k2 * h / 2, *par)
        k4 = f(x + k3 * h, *par)
        x = x + (k1 + 2 * k2 + 2 * k3 + k4) * h / 6
        j += 1
    return x

def get_rk4(f, nx, nu, casaditype="SX", Delta=1, M=1):
    if casaditype=="SX":
        x = cs.SX.sym("x", nx)
        u = cs.SX.sym("u", nu)
    elif casaditype=="MX":
        x = cs.MX.sym("x", nx)
        u = cs.MX.sym("u", nu)
    else:
        raise ValueError("Wrong CasADi type.")

    f = cs.Function("f", [x, u], [cs.vertcat(*f(x, u))])
    f_rk4 = rk4(f, x, [u], Delta=Delta, M=M)
    f_casadi = cs.Function("f_rk4", [x, u], [f_rk4])
    return f_casadi