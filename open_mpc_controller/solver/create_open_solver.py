import casadi as cs
import opengen as og
import matplotlib.pyplot as plt
import numpy as np
import time
import util

useSX = True  # Use CasADi SX or MX. Seems quicker with SX
doSim = False  # Run simulation after creating the solver

N = 10
NX = 3
NU = 2
sampling_time = 0.1
NSim = 100

Q = cs.DM.eye(NX) * [1.0, 1.0, 0.0001]
R = cs.DM.eye(NU) * [0.1, 50.0]
QN = cs.DM.eye(NX) * [10.0, 10.0, 0.0001]


def dynamics_ct(_x, _u):
    return [_u[0] * cs.cos(_x[2]),
                        _u[0] * cs.sin(_x[2]),
                        _u[1]]


def stage_cost(_x, _u, _x_ref=None, _u_ref=None):
    if _x_ref is None:
        _x_ref = cs.DM.zeros(_x.shape)
    if _u_ref is None:
        _u_ref = cs.DM.zeros(_u.shape)
    dx = _x - _x_ref
    du = _u - _u_ref
    return cs.mtimes([dx.T, Q, dx]) + cs.mtimes([du.T, R, du])


def terminal_cost(_x, _x_ref=None):
    if _x_ref is None:
        _x_ref = cs.DM.zeros(_x.shape)
    dx = _x - _x_ref
    return cs.mtimes([dx.T, QN, dx])

if useSX:
    x_0 = cs.SX.sym("x_0", NX)
    x_ref = cs.SX.sym("x_ref", NX)
    u_k = [cs.SX.sym('u_' + str(i), NU) for i in range(N)]
    dynamics_dt = util.get_rk4(dynamics_ct, NX, NU, casaditype="SX", Delta=sampling_time)
else:
    x_0 = cs.MX.sym("x_0", NX)
    x_ref = cs.MX.sym("x_ref", NX)
    u_k = [cs.MX.sym('u_' + str(i), NU) for i in range(N)]
    dynamics_dt = util.get_rk4(dynamics_ct, NX, NU, casaditype="MX", Delta=sampling_time)


x_t = x_0
total_cost = 0

for t in range(0, N):
    total_cost += stage_cost(x_t, u_k[t], x_ref)  # update cost
    x_t = dynamics_dt(x_t, u_k[t])  # update state

total_cost += terminal_cost(x_t, x_ref)  # terminal cost

optimization_variables = []
optimization_parameters = []

optimization_variables += u_k
optimization_parameters += [x_0]
optimization_parameters += [x_ref]

optimization_variables = cs.vertcat(*optimization_variables)
optimization_parameters = cs.vertcat(*optimization_parameters)

umin = [-2.0, -1.0] * N  # - cs.DM.ones(NU * N) * cs.inf
umax = [2.0, 1.0] * N  # cs.DM.ones(NU * N) * cs.inf

bounds = og.constraints.Rectangle(umin, umax)

problem = og.builder.Problem(optimization_variables, optimization_parameters, total_cost).with_constraints(bounds)

if doSim:
    build_config = og.config.BuildConfiguration()\
        .with_build_directory("optimization_engine")\
        .with_build_mode("release")\
        .with_build_c_bindings()\
        .with_tcp_interface_config()
else:
    build_config = og.config.BuildConfiguration()\
        .with_build_directory("optimization_engine")\
        .with_build_mode("release")\
        .with_build_c_bindings()
meta = og.config.OptimizerMeta()\
    .with_optimizer_name("mpc_controller")
solver_config = og.config.SolverConfiguration()\
    .with_tolerance(1e-5)
builder = og.builder.OpEnOptimizerBuilder(problem,
                                          meta,
                                          build_config,
                                          solver_config) \
    .with_verbosity_level(1)
builder.build()

if doSim:
    # Use TCP server
    # ------------------------------------
    mng = og.tcp.OptimizerTcpManager('optimization_engine/mpc_controller')
    mng.start()

    current_x_0 = [-1, -1, 0]
    current_x_ref = [2, 2, 0]
    current_u = [0, 0]
    cur_opt_var = [0] * (N * NU)  # current optimization variables
    cur_opt_par = current_x_0 + current_x_ref  # current optimization parameters
    average_solvetime = 0.0

    x_sim = np.zeros((NSim + 1, NX))
    x_sim[0:N + 1] = current_x_0

    for k in range(0, NSim):

        solver_status = mng.call(cur_opt_par)
        cur_opt_var = solver_status['solution']
        solvetime = solver_status['solve_time_ms']
        current_u = cur_opt_var[0:NU]
        current_x_0 = dynamics_dt(current_x_0, current_u).toarray().flatten()
        x_sim[k + 1] = current_x_0

        print(("Step %d/%d took %f [ms]") % (k, NSim, solvetime))
        average_solvetime += solvetime

        cur_opt_par = list(current_x_0) + current_x_ref

    mng.kill()

    print(("Average solvetime %.4f [ms]") % (average_solvetime/NSim))

    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(x_sim[:, 0])
    plt.subplot(3, 1, 2)
    plt.plot(x_sim[:, 1])
    plt.subplot(3, 1, 3)
    plt.plot(x_sim[:, 0], x_sim[:, 1])
    plt.show()
