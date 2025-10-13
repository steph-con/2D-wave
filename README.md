# 2D-WAVE SIMULATION

Two-dimensional wave simulation in python.

## Context

A square pond of length 10 by 10 is subject to a force that produces a ripple on its surface. The aim is to simulate the effect of this disturbance on the fluid, namely to estimate the height of the fluid at each point in the pond.

With $u(x, y, t)$ as the water level of the pond at coordinates $(x, y)$ at time $t$, the wave equation of the surface of the pond can be described as:

$$
\frac{\partial^2 u}{\partial t^2} = c^2 \left(\frac{\partial^2 u}{\partial x^2} +
\frac{\partial^2 u}{\partial y^2} \right) - \nu \frac{\partial u}{\partial t}
$$

- $c = 1 \rightarrow$ Wave constant
- $\nu = 0.002 \rightarrow$ Friction factor

### Disturbance

There is a sinusoidal disturbance in the middle of the pond during the first 2.5 seconds given by:

$$u(5,5,t) = \sin\left(\frac{t}{10}\right) , \quad \forall t \in (0,2.5) $$

### Initial conditions

$$u(x, y, 0) = 0, \quad \forall x \in (0, 10), y \in (0, 10)$$

### Dirichlet boundary conditions

$$
\begin{align*}
    u(0, y, t) &  = 0, \quad \forall t, y \\
    u(10, y, t) & = 0, \quad \forall t, y \\
    u(x, 0, t) &  = 0, \quad \forall t, x \\
    u(x, 10, t) & = 0, \quad \forall t, x
\end{align*}
$$

## To-do

- [x] Add result files for a 20s simulation
- [ ] Add functionality for energy calculations: Check energy conservation
- [ ] Add Jupyter Notebook file explaining the theory behind the solution
- [ ] Switch to using qt backend and add capability for frame navigation