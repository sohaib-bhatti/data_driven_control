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

    t = np.linspace(0, 3, num=1000)

    t, y = ct.impulse_response(G, t)
    """fig1, ax1 = plt.subplots()
    ax1.plot(t, y)
    plt.xlabel('Time')
    plt.ylabel('Response')
    plt.title("AFM Impulse Response")
    plt.show()"""

    mco = int(np.floor((y.shape[0]-1)/2))

    Ar, Br, Cr, Dr, HSVs = ERA(y, mco, mco, 10)

    sys_ERA = ct.ss(Ar, Br, Cr, Dr, 1)
    mag, phase, omega = ct.bode([G, sys_ERA])
    plt.tight_layout()

    ax1, ax2 = plt.gcf().axes     # get subplot axes

    plt.sca(ax1)                 # magnitude plot
    plt.legend(["Transfer function", "ERA"])
    plt.title("Gain")

    plt.sca(ax2)                 # phase plot
    plt.title("Frequency")
    plt.show()


# ERA code adapted from Steve Brunton
def ERA(Y, m, n, r):
    Dr = Y[0]

    H = np.zeros((m, n))
    H2 = np.zeros((m, n))

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
    print(Y)
    print(H)
    return Ar, Br, Cr, Dr, HSVs


if __name__ == '__main__':
    main()