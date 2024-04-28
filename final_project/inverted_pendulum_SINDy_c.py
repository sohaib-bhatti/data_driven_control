import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pysindy as ps
import do_mpc
import casadi


def main():
    t_start = 0
    t_stop = 16
    x0 = [0, 0.3, 0, 1]
    dt = 1
    t = np.arange(t_start, t_stop, dt)

    x = odeint(diff, x0, t)

    print("diff done!")

    x_sindy, x_dot_sindy, model = sindy(x[:, :2], t)

    sim = model.simulate(x0, t=t)

    model.print()
    print("coefficients: %s", model.coefficients().T)

    plt.figure(1)

    # plot displacement
    plt.subplot(4, 1, 1)
    plt.plot(t, x[:, 0])
    plt.plot(t, sim[:, 0], linestyle='dashed')

    plt.ylabel('x (m)')
    plt.legend(["x", "x_sindy"])
    plt.grid()

    # plot angle
    plt.subplot(4, 1, 2)
    plt.plot(t, x[:, 1])
    plt.plot(t, sim[:, 1], linestyle='dashed')

    plt.ylabel(("theta (rad)"))
    plt.legend(["theta", "theta_sindy"])
    plt.grid()

    # plot velocity
    plt.subplot(4, 1, 3)
    plt.plot(t, x[:, 2])
    plt.plot(t, sim[:, 2], linestyle='dashed')

    plt.ylabel('x_dot (rad)')
    plt.legend(["x_dot", "x_dot_sindy"])
    plt.grid()

    # plot angular velocity
    plt.subplot(4, 1, 4)
    plt.plot(t, x[:, 3])
    plt.plot(t, sim[:, 3], linestyle='dashed')

    plt.xlabel('t (s)')
    plt.ylabel('theta_dot (rad/s)')
    plt.legend(["theta_dot", "theta_dot_sindy"])
    plt.grid()

    plt.show()


def diff(x, t):
    # dynamics obtained from Brunton
    M = 1  # mass of cart, kg
    m = 1  # mass of pendulum, kg
    m_t = M + m
    le = 0.2  # length of pendulum, m
    g = 9.8  # gravity, m/s^2

    u = -0.2 + 0.5 * np.sin(6*t)

    dxdt = x[2]
    dTdt = x[3]
    dx_dotdt = (m*le**2*np.sin(x[1])*x[3]**2 + u*le +
                m*g*np.sin(x[1])*np.cos(x[1])) /\
        (le*(m_t - m*np.cos(x[1])**2))
    dT_dotdt = -(m_t*g*np.sin(x[1]) + u*le*np.cos(x[1]) +
                 m*le**2*np.sin(x[1]) * np.cos(x[1])*x[3]**2) /\
        (le**2*(m_t - m*np.cos(x[1])**2))

    return [dxdt, dTdt, dx_dotdt, dT_dotdt]


def sindy(x, t):
    optimizer = ps.optimizers.STLSQ(threshold=1e-4,
                                    max_iter=20000)

    differentiation_method = ps.differentiation.FiniteDifference()
    # pylint: disable=protected-access
    x_dot = differentiation_method._differentiate(x, t)

    functions = [lambda x: np.sin(x)**-2,
                 lambda x: np.cos(x)**-2,
                 ]
    lib_custom = ps.CustomLibrary(library_functions=functions)

    feature_library = ps.ConcatLibrary([ps.FourierLibrary(),
                                        ps.PolynomialLibrary(3),
                                        lib_custom])

    model = ps.SINDy(optimizer=optimizer,
                     differentiation_method=differentiation_method,
                     feature_library=feature_library,
                     feature_names=["x", "theta", "x_dot", "theta_dot"],
                     )

    print("model created!")

    print(x)
    print(t)
    model.fit(x, t=t, ensemble=True)

    return x, x_dot, model


def mpc_ss(ss, x0, ts, num_iters):
    model = do_mpc.model.Model('continuous')

    x = model.set_variable(var_type='_x', var_name='x', shape=(2, 1))
    u = model.set_variable(var_type='_u', var_name='u')

    A = np.vstack((ss[1], ss[2]))
    print(A)
    print("here!")
    B = np.array([0, 1])

    x_next = A@x + B@u

    model.set_rhs('x', x_next)

    model.set_expression(expr_name='cost', expr=casadi.sum1(x**2))

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

    mpc.set_rterm(u=1e-4)  # input penalty

    # set bounds
    mpc.bounds['lower', '_x', 'x'] = -20
    mpc.bounds['upper', '_x', 'x'] = 20
    mpc.bounds['lower', '_u', 'u'] = -20
    mpc.bounds['upper', '_u', 'u'] = 20

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

    x0 = np.array([5, -1])

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
