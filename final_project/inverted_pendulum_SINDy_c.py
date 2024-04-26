import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pysindy as ps
import do_mpc
import casadi


def main():
    t_start = 0
    t_stop = 60
    x0 = [2, -2, 3, 0.2]
    num_samples = 1000
    t = np.linspace(t_start, t_stop, num=num_samples)

    ts = (t_stop-t_start)/num_samples

    x = odeint(diff, x0, t)

    print("diff done!")

    x_sindy, x_dot_sindy, model = sindy(x, t)

    sim = model.simulate(x0, t=t)

    model.print()
    print("coefficients: %s", model.coefficients().T)

    plt.figure(1)

    plt.subplot(2, 1, 1)
    plt.plot(t, x[:, 0])
    plt.plot(t, sim[:, 0], linestyle='dashed')

    plt.title('cart displacement')
    plt.xlabel('t (s)')
    plt.ylabel('x (m)')
    plt.legend(["x", "x_sindy"])
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(t, x[:, 1])
    plt.plot(t, sim[:, 1], linestyle='dashed')

    plt.title('pendulum angle')
    plt.xlabel('t (s)')
    plt.ylabel('theta (rad)')
    plt.legend(["theta", "theta_sindy"])
    plt.grid()

    plt.show()

def diff(x, t):
    # dynamics obtained from Feedback Systems by Astrom and Murray
    M = 2  # mass of cart, kg
    m = 0.3  # mass of pendulum, kg
    m_t = M + m
    c = 0.1  # damping coefficients, N s/m
    gamma = 0.1
    l = 0.2  # length of pendulum, m
    g = 10  # gravity, m/s^2
    J = 1  # moment of inertia kg m^2
    J_t = J + m * l**2

    u = 0

    dxdt = x[2]
    dTdt = x[3]
    dx_dotdt = (-m*l**2*x[3] + m*g*(m*l**2/J_t)*np.sin(x[1])*np.cos(x[1]) - c*x[2]-gamma*l*m*np.cos(x[1])*x[3] + u) /\
        (m_t - m*(m*l**2/J_t)*np.cos(x[1])**2)
    dT_dotdt = (-m*l**2*np.sin(x[1])*np.cos(x[1])*x[3]**2 + m_t*g*l*np.sin(x[1]) - c*l*np.cos(x[1])*x[2] - gamma*(m_t/m)*x[3] + l*np.cos(x[1])*np.cos(x[1])*u) /\
        (J_t*(m_t/m) - m*(l*np.cos(x[1])**2))

    dxdt = [dxdt, dTdt, dx_dotdt, dT_dotdt]
    return dxdt


def sindy(x, t):
    optimizer = ps.optimizers.STLSQ(threshold=1e-8,
                                    max_iter=20000)

    differentiation_method = ps.differentiation.FiniteDifference()
    # pylint: disable=protected-access
    x_dot = differentiation_method._differentiate(x, t)

    feature_library = ps.FourierLibrary() + ps.PolynomialLibrary(2)

    model = ps.SINDy(optimizer=optimizer,
                     differentiation_method=differentiation_method,
                     feature_library=feature_library,
                     feature_names=["x", "theta", "x_dot", "theta_dot"],
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
