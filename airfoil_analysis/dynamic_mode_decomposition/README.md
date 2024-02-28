# Background
Dynamic mode decomposition (DMD) is a discrete-time algorithm used to decompose data into modes that give information about the spatial behavior of the system along with how it evolves in time. The problem to solve to find the linear operator $\mathbf{A}$ to relate two snapshots in time $\mathbf{X}$ and $\mathbf{X'}$:
$$\mathbf{X'} \approx \mathbf{AX}$$

We will conduct a DMD on Scott Dawson's pitching airfoil CFD data to obtain spatial modes and important temporal data.

# Algorithm
Dawson has an implementation of a DMD included with the airfoil data. The algorithm is as follows[^1]:

1. Conduct a singular value decomposition (SVD) on X
$$\mathbf{X} \approx \mathbf{\tilde{U}\tilde{\Sigma}\tilde{V}^*}$$
2. Compute the pseudo-inverse of $\mathbf{X}$ to find $\mathbf{A}$
$$\mathbf{A} = \mathbf{X'\tilde{V}\tilde{\Sigma}^{-1}\tilde{U}^\*}$$
$$\mathbf{\tilde{A}} = \mathbf{\tilde{U}^\*A\tilde{U}} = \mathbf{\tilde{U}^\*X'\tilde{V}\tilde{\Sigma}^{-1}}$$
3. Find the eigendecomposition of $\mathbf{\tilde{A}}$
$$\mathbf{\tilde{A}W} = \mathbf{W\Lambda}$$
4. Find the DMD modes
$$\mathbf{\phi} = \mathbf{X'\tilde{V}\tilde{\Sigma}^{-1}W}$$

Dawson's implementation returns $\mathbf{\Phi}$, $\mathbf{\Lambda}$, $\mathbf{\tilde{A}}$, and the amplitudes.

[^1]: Brunton, S. L., & Kutz, J. N. (2022). Data-driven science and engineering machine learning, Dynamical Systems, and Control. Cambridge University Press. 

# Analysis
A DMD was conducted on a data matrix with the x and y velocities of the flow field. It resulted in the following graphs.

![alt text](.images/frequencies.png)
