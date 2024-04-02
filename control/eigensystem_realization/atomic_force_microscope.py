import control as ct
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power


def main():
    # define dynamics for atomic force microscope, from Astrom and Murray
    f1 = 2.4  # khz
    f2 = 2.6  # khz
    f3 = 6.5  # khz
    f4 = 8.3  # khz
    f5 = 9.3  # khz
    z1 = 0.03
    z2 = 0.03
    z3 = 0.042
    z4 = 0.025
    z5 = 0.032
    w1 = 2 * np.pi * f1
    w2 = 2 * np.pi * f2
    w3 = 2 * np.pi * f3
    w4 = 2 * np.pi * f4
    w5 = 2 * np.pi * f5
    k = 5
    tau = 10**-4  # s

    s = ct.TransferFunction.s
    num_delay, den_delay = ct.pade(tau, 1)

    G = (k * w2**2 * w3**2 * w5**2 *
         (s**2 + 2*z1*w1 + w1**2) *
         (s**2 + 2*z4*w4*s + w4**2) *
         ct.TransferFunction(num_delay, den_delay)) /\
        (w1**2 * w4**2 *
         (s**2 + 2*z2*w2*s + w2**2) *
         (s**2 + 2*z3*w3*s + w3**2) *
         (s**2 + 2*z5*w5*s + w5**2))

    start_time = 0
    end_time = 3
    num_steps = 3001
    t = np.linspace(start_time, end_time, num=num_steps)
    dt = (end_time - start_time) / (num_steps-1)

    # impulse response of G for ERA to use
    t_i, y = ct.impulse_response(G, t)
    # add Gaussian noise to impulse response
    noise = True
    if noise:
        y += np.random.normal(0, np.amax(y) * 0.01, y.shape[0])

    plt.figure(1)

    plt.plot(t_i, y)
    plt.xlabel('Time')
    plt.ylabel('Response')

    if noise:
        plt.title("Noisy AFM Impulse Response")
    else:
        plt.title("AFM Impulse Response")

    mco = int(np.floor((y.shape[0]-1)/2))  # dimension for Hankel matrix
    ss_size = 10

    # obtain state space realization from ERA
    Ar, Br, Cr, Dr, HSVs = ERA(y, mco, mco, ss_size)

    # construct system from ERA and plot frequency response for both systems
    sys_ERA = ct.ss(Ar, Br, Cr, Dr, dt) * dt
    t_i2, y_ERA = ct.impulse_response(sys_ERA, t)
    plt.plot(t_i2, y_ERA)
    plt.legend(["Transfer function", "ERA"])

    plt.figure(2)
    mag, phase, omega = ct.bode([G, sys_ERA])
    plt.tight_layout()

    ax1, ax2 = plt.gcf().axes  # get subplot axes

    plt.sca(ax1)  # magnitude plot
    plt.legend(["Transfer function", "ERA"])
    plt.title("Gain")

    plt.sca(ax2)  # phase plot
    plt.title("Frequency")
    plt.xlabel("Frequency (krads/s)")

    # plot the singular values to find appropriate rank for ERA
    plt.figure(3)
    plt.stem(HSVs)
    plt.xlim([-1, 20])
    plt.xlabel("Index")
    plt.ylabel("Singular Value")
    plt.title("Singular Values of the Hankel Matrix")

    plt.show()


# ERA code adapted from Steve Brunton
def ERA(Y, m, n, r):
    Dr = Y[0]

    H = np.zeros((m, n))
    H2 = np.zeros((m, n))

    # construct Hankel matrices
    for i in range(m):
        for j in range(n):
            H[i, j] = Y[i + j]
            H2[i, j] = Y[i + j + 1]

    U, S, VT = np.linalg.svd(H, full_matrices=0)
    V = VT.T
    Sigma = np.diag(S[:r])
    Ur = U[:, :r]
    Vr = V[:, :r]
    Ar = fractional_matrix_power(Sigma, -0.5) @ Ur.T @ H2 @ Vr @ \
        fractional_matrix_power(Sigma, -0.5)
    Br = fractional_matrix_power(Sigma, -0.5) @ Ur.T @ H[:, :1]
    Cr = H[:1, :] @ Vr @ fractional_matrix_power(Sigma, -0.5)
    HSVs = S
    return Ar, Br, Cr, Dr, HSVs


if __name__ == '__main__':
    main()
