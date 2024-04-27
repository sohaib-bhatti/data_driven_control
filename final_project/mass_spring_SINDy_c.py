import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pysindy as ps
import do_mpc


def main():
    t_start = 0
    t_stop = 100
    x0 = [5, -1]
    num_samples = 1000
    t = np.linspace(t_start, t_stop, num=num_samples)

    ts = (t_stop-t_start)/num_samples

    x = odeint(diff, x0, t)

    x_sindy, x_dot_sindy, model = sindy(x, t)

    sim = model.simulate(x0, t=t)

    model.print()
    print("coefficients: %s", model.coefficients().T)

    state_CL, u_CL = mpc_ss(model.coefficients().T, x0, ts, num_samples)
    state_CL_original = odeint(diff_CL, (x0, u_CL), t)

    x_CL = state_CL[:, 0]
    x_CL_original = state_CL_original[:, 0]

    plt.figure(1)

    plt.plot(t, x[:, 0])
    plt.plot(t, sim[:, 0], linestyle='dashed')

    plt.title('mass spring displacement')
    plt.xlabel('t (s)')
    plt.ylabel('x (m)')
    plt.legend(["x", "x_sindy"])
    plt.grid()

    plt.figure(2)

    plt.plot(t, x[:, 1])
    plt.plot(t, sim[:, 1], linestyle='dashed')

    plt.title('mass spring velocity')
    plt.xlabel('t (s)')
    plt.ylabel('x_dot (m/s)')
    plt.legend(["x_dot", "x_dot_sindy"])
    plt.grid()

    plt.figure(3)

    plt.subplot(2, 1, 1)
    plt.plot(t, x_CL)
    plt.plot(t, x_CL_original)
    plt.title('closed loop')
    plt.xlabel('t (s)')
    plt.ylabel('x (m)')
    plt.legend(["x_sindy", "x_original"])
    plt.ylim([-6, 6])

    plt.subplot(2, 1, 2)
    plt.plot(t, u_CL)
    plt.title('input force')
    plt.xlabel('t (s)')
    plt.ylabel('f (N/kg)')
    plt.legend(["u"])
    plt.grid()

    plt.show()


def diff(x, t):
    c = 0.1  # Ns/m
    k = 8  # N/m
    m = 2  # kg
    F = 0
    dx1dt = x[1]
    dx2dt = (F - c*x[1] - k*x[0])/m

    dxdt = [dx1dt, dx2dt]
    return dxdt


def diff_CL(x, t, u):
    c = 0.1  # Ns/m
    k = 8  # N/m
    m = 2  # kg
    F = u*m
    dx1dt = x[1]
    dx2dt = (F - c*x[1] - k*x[0])/m

    dxdt = [dx1dt, dx2dt]
    return dxdt


def sindy(x, t):
    optimizer = ps.optimizers.STLSQ(threshold=1e-8,
                                    max_iter=20000)

    differentiation_method = ps.differentiation.FiniteDifference()
    # pylint: disable=protected-access
    x_dot = differentiation_method._differentiate(x, t)

    model = ps.SINDy(optimizer=optimizer,
                     differentiation_method=differentiation_method,
                     feature_names=["x", "x_dot"],
                     discrete_time=False)
    model.fit(x, t=t, ensemble=True)

    return x, x_dot, model


def mpc(ss):
    model = do_mpc.model.Model('continuous')

    x = model.set_variable(var_type='_x', var_name='x', shape=(1, 1))
    x_dot = model.set_variable(var_type='_x', var_name='x_dot', shape=(1, 1))

    f = model.set_variable(var_type='_u', var_name='f')

    model.set_rhs('dx', ss[1][0]*x + ss[1][1]*x_dot)
    model.set_rhs('dx_dot', ss[2][0]*x + ss[2][1]*x_dot + f)

    model.setup()

    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {'n_horizon': 20,
                 't_step': 0.1,
                 'n_robust': 1,
                 'store_full_solution': True, }
    mpc.set_param(**setup_mpc)


def mpc_ss(ss, x0, ts, num_iters):
    model = do_mpc.model.Model('continuous')

    x = model.set_variable(var_type='_x', var_name='x', shape=(2, 1))
    u = model.set_variable(var_type='_u', var_name='u')

    A = np.vstack((ss[1], ss[2]))
    B = np.array([0, 1])

    x_next = A@x + B@u

    model.set_rhs('x', x_next)

    model.set_expression(expr_name='cost', expr=(x[0]+2)**2)

    model.setup()

    mpc = do_mpc.controller.MPC(model)
    mpc.settings.supress_ipopt_output()

    setup_mpc = {'n_horizon': 20,
                 't_step': 0.1,
                 'n_robust': 1,
                 'store_full_solution': True, }
    mpc.set_param(**setup_mpc)

    mterm = model.aux['cost']  # terminal cost
    lterm = model.aux['cost']  # terminal cost
    # stage cost

    mpc.set_rterm(u=1e-1)  # input penalty

    # set bounds
    mpc.bounds['lower', '_x', 'x'] = -20
    mpc.bounds['upper', '_x', 'x'] = 20
    mpc.bounds['lower', '_u', 'u'] = -2
    mpc.bounds['upper', '_u', 'u'] = 2

    mpc.set_objective(mterm=mterm, lterm=lterm)

    mpc.setup()
    estimator = do_mpc.estimator.StateFeedback(model)
    simulator = do_mpc.simulator.Simulator(model)

    params_simulator = {'integration_tool': 'idas',
                        'abstol': 1e-8,
                        'reltol': 1e-8,
                        't_step': ts}

    simulator.set_param(**params_simulator)
    simulator.setup()

    x0 = np.array([5, -5])

    mpc.x0 = x0
    simulator.x0 = x0
    estimator.x0 = x0

    mpc.set_initial_guess()

    for k in range(num_iters):
        u0 = mpc.make_step(x0)
        y_next = simulator.make_step(u0)
        x0 = estimator.make_step(y_next)

    return mpc.data['_x'], mpc.data['_u']


if __name__ == '__main__':
    main()
