import h5py
import numpy as np
import matplotlib.pyplot as plt

param_file = "Airfoil Analysis/airfoilDNS_parameters.h5"
grid_file = "Airfoil Analysis/airfoilDNS_grid.h5"
data_file = "Airfoil Analysis/airfoilDNS_a25f0p35.h5"


with h5py.File(param_file,'r+') as f:
    dt_field = np.squeeze(f['/dt_field'][()]) # timestep for field variables (velocity and vorticity)
    dt_force = np.squeeze(f['/dt_force'][()]) # timestep for scalar quantities
    Re = np.squeeze(f['/Re'][()])
    FreqsAll = np.squeeze(f['/frequencies'][()]) # pitching frequencies
    alpha_p = np.squeeze(f['/alpha_p'][()]) # pitching amplitude (deg)
    alpha_0s = np.squeeze(f['/alpha_0s'][()]) # base angles of attack (deg) (25 and 30)
    pitch_axis = np.squeeze(f['/pitch_axis'][()]) # 0.5, midchord pitching


BaseAngle = 30
freq = 0.35
tstep = 1

with h5py.File(grid_file,'r+') as f:
    x = np.squeeze(f['/x'][()])
    y = np.squeeze(f['/y'][()])
    nx = np.squeeze(np.size(x))
    ny = np.squeeze(np.size(y))

with h5py.File(data_file, 'r+') as f:
    ux = np.squeeze(f['/ux'][()])
    uy = np.squeeze(f['/uy'][()])
    xa = np.squeeze(f['/ux'][()])
    ya = np.squeeze(f['/uy'][()])
    t_field = np.squeeze(f['/t_field'][()])
    t_force = np.squeeze(f['/t_force'][()])
    nt = np.squeeze(np.shape(t_field))

uxreshape = np.reshape(ux, (nx*ny, nt), order='F')
uyreshape = np.reshape(uy, (nx*ny, nt), order='F')

data = np.vstack((uxreshape, uyreshape))

data_mean = np.mean(data, axis=1)
data = data - np.reshape(data_mean, (len(data_mean), 1)) * np.ones([1, nt], order='F')

try:
    U = np.load('Airfoil Analysis/svd_u.npy')
    S = np.load('Airfoil Analysis/svd_s.npy')
    V = np.load('Airfoil Analysis/svd_v.npy')
except FileNotFoundError:
    U, S, V = np.linalg.svd(data, full_matrices=False)
    np.save('Airfoil Analysis/svd_u.npy', U)
    np.save('Airfoil Analysis/svd_s.npy', S)
    np.save('Airfoil Analysis/svd_v.npy', V)


eigs = np.square(S)

fig, ax = plt.subplots()
plt.yscale("log")
ax.plot(np.arange(len(eigs)), eigs)
plt.ylim(1, 10**6)
ax.set_title('Singular Values')

half = int(np.shape(U)[0]/2)

num_x_plots = 3
num_y_plots = 2

num_plots = num_x_plots * num_y_plots


fig2, axes = plt.subplots(num_x_plots, num_y_plots)

X, Y = np.meshgrid(x, y)

for i in range(num_plots):
    row, col = i // 2, i % 2
    ax = axes[row, col]
    Ux_field = np.transpose(np.reshape(U[half:,i], (nx, ny), order='F'))
    print(np.shape(U[half:,i]))
    ax.contour(X, Y, Ux_field)

fig3, axes2 = plt.subplots(num_x_plots, num_y_plots)
for i in range(num_plots):
    row, col = i // 2, i % 2
    ax = axes2[row, col]
    Uy_field = np.transpose(np.reshape(U[:half,i], (nx, ny), order='F'))
    ax.contour(X, Y, Uy_field)

plt.show()