import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pysindy as ps


def main():
    t_start = 0
    t_stop = 200
    x_0 = [10, -1]
    num_samples = 100000
    t = np.linspace(t_start, t_stop, num=num_samples)

    x = odeint(diff, x_0, t)

    # Plot the Results
    plt.plot(t, x[:, 0])
    plt.title('mass spring')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.legend(["x"])
    plt.grid()
    plt.show()


def diff(x, t):
    c = 4
    k = 2
    m = 20
    F = 0.5 * np.sin(t)

    dx1dt = x[1]
    dx2dt = (F - c*x[1] - k*x[0])/m

    dxdt = [dx1dt, dx2dt]
    return dxdt


def sindy(x, t):
    # Functions to be applied to the data x
    functions = [lambda x: np.exp(x),
                 lambda x: np.sin(x),
                 lambda x: np.cos(x)]

    # Functions to be applied to the data x_dot
    x_dot_functions = [lambda x: np.exp(x),
                       lambda x: np.sin(x),
                       lambda x: np.cos(x)]

    # library function names includes both
    # the x_library_functions and x_dot_library_functions names
    function_names = [lambda x: "exp(" + x + ")",
                      lambda x: "sin(" + x + ")",
                      lambda x: "cos(" + x + ")",
                      lambda x: "exp(" + x + ")",
                      lambda x: "sin(" + x + ")",
                      lambda x: "cos(" + x + ")",]

    lib = ps.SINDyPILibrary(library_functions=functions,
                            x_dot_library_functions=x_dot_functions,
                            function_names=function_names,
                            t=t).fit(x)

    lib.transform(x)
    print("With function names: ")
    print(lib.get_feature_names(), "\n")

    sindy_library = ps.ODELibrary(
        library_functions=functions,
        temporal_grid=t,
        function_names=function_names,
        include_bias=True,
        implicit_terms=True,
        derivative_order=1
    )


if __name__ == '__main__':
    main()