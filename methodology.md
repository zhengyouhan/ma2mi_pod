# Methodology

## 1. Car-Following Dynamics

The microscopic traffic dynamics are governed by the Intelligent Driver Model (IDM) (Treiber et al., 2000). The acceleration of vehicle $i$ at time $t$ is given by

$$
a_i(t) = a_\text{max} \left[ 1 - \left(\frac{v_i(t)}{v_0}\right)^\delta - \left(\frac{s^*(v_i, \Delta v_i)}{s_i(t)}\right)^2 \right],
$$

where $v_i$ denotes the current speed, $s_i = x_{\text{leader}(i)} - x_i$ is the bumper-to-bumper gap to the preceding vehicle, and $\Delta v_i = v_i - v_{\text{leader}(i)}$ is the approaching rate. The desired gap $s^*$ is defined as

$$
s^*(v_i, \Delta v_i) = s_0 + v_i T_i + \frac{v_i \Delta v_i}{2\sqrt{a_\text{max} \, b}},
$$

combining a minimum standstill distance $s_0$, a speed-dependent safe following distance scaled by the time headway $T_i$, and an interaction term that increases the desired gap when the ego vehicle is faster than its leader. The parameter roles are summarized in Table 1.

**Table 1.** IDM parameters and their roles in the reconstruction framework.

| Parameter | Description | Status |
|-----------|-------------|--------|
| $v_0$ | Desired free-flow speed | Learnable (global) |
| $a_\text{max}$ | Maximum acceleration | Learnable (global) |
| $T_i$ | Time headway | Learnable (per-vehicle) |
| $b$ | Comfortable deceleration | Fixed (2.0 m/s$^2$) |
| $s_0$ | Minimum standstill gap | Fixed (2.0 m) |
| $\delta$ | Free-flow exponent | Fixed (4) |

The simulation domain consists of a multi-lane road segment in which vehicles are assigned to lanes at initialization and maintain their lane assignment throughout. Within each lane, vehicles form an ordered leader-follower chain. The leading vehicle in each lane follows a virtual ghost driver that travels at a constant speed $v_\text{ghost}$ and provides the upstream boundary condition. The trajectories are integrated using a forward Euler scheme with time step $\Delta t = 0.1$ s:

$$
v_i(t + \Delta t) = v_i(t) + a_i(t) \, \Delta t, \qquad x_i(t + \Delta t) = x_i(t) + v_i(t + \Delta t) \, \Delta t.
$$

The entire $N_T$-step rollout is implemented as a single unrolled computation graph in PyTorch, enabling exact gradient computation via reverse-mode automatic differentiation. Three differentiable safeguards preserve gradient flow at physical constraints: (i) acceleration bounds are enforced through softplus-based soft clamping rather than hard truncation, retaining nonzero gradients at the constraint boundary; (ii) the gap $s_i$ is smoothly bounded below via a softplus function to prevent numerical divergence in the $s^*/s_i$ interaction term; and (iii) non-negative speeds are maintained by imposing $a_i \geq -v_i / \Delta t$, which guarantees $v_i + a_i \Delta t \geq 0$ without introducing zero-gradient plateaus.


## 2. Parameterization

The reconstruction problem seeks to recover the heterogeneous time headway profile $\{T_i\}_{i=1}^N$ together with the global parameters $v_0$ and $a_\text{max}$ from macroscopic detector observations. All remaining IDM parameters, the initial conditions $(x_i(0), v_i(0))$, the lane assignments, and the leader-follower topology are treated as known and held fixed.

Each bounded parameter $\theta \in [\ell, h]$ is represented through a sigmoid reparameterization. An unconstrained variable $u \in \mathbb{R}$ is introduced and mapped to the physical domain via

$$
\theta(u) = \ell + (h - \ell) \, \sigma(u),
$$

where $\sigma(\cdot)$ denotes the standard logistic function. The inverse mapping used for initialization is $u = \log[(\theta - \ell)/(h - \theta)]$. This reparameterization ensures exact satisfaction of the box constraints while maintaining smooth, nonzero gradients throughout the feasible region, including near the boundaries where hard clamping would produce zero gradients. The bounds adopted in this work are $T_i \in [0.8, 2.5]$ s, $v_0 \in [15, 40]$ m/s, and $a_\text{max} \in [0.5, 3.0]$ m/s$^2$.

All per-vehicle headway parameters are initialized at a common prior mean $T_\text{mean}$. The optimizer must therefore discover the heterogeneous $T_i$ profile solely from the information contained in the macroscopic observations. The global parameters $v_0$ and $a_\text{max}$ are initialized at their respective nominal values from the scenario configuration. Parameter updates are performed using the Adam optimizer (Kingma and Ba, 2015) with learning rate $\eta = 0.05$ and default momentum parameters ($\beta_1 = 0.9$, $\beta_2 = 0.999$). Convergence is typically observed within 60--80 iterations, and all experiments in this work use 100 iterations.


## 3. Observation Operator: Micro-to-Macro Aggregation

A differentiable observation operator is required to map the microscopic state $\{x_i(t), v_i(t)\}$ to macroscopic quantities comparable with inductive loop detector data. This operator constitutes the forward model through which gradients propagate from the macroscopic loss back to the microscopic parameters.

### 3.1 Gaussian kernel density estimation

For each lane $\ell$, the macroscopic density, speed, and flow fields are computed on a spatial grid $\{x_m\}_{m=1}^M$ via Gaussian kernel smoothing. The density field is

$$
\rho_\ell(x, t) = \frac{1}{\sqrt{2\pi}\,\sigma} \sum_{i \in \mathcal{V}_\ell} \exp\!\left(-\frac{(x - x_i(t))^2}{2\sigma^2}\right),
$$

where $\mathcal{V}_\ell$ denotes the set of vehicles in lane $\ell$ and $\sigma$ is the kernel bandwidth. The macroscopic speed field is computed as a kernel-weighted average:

$$
u_\ell(x, t) = \frac{\sum_{i \in \mathcal{V}_\ell} w_i(x, t) \, v_i(t)}{\sum_{i \in \mathcal{V}_\ell} w_i(x, t) + \varepsilon}, \qquad w_i(x, t) = \exp\!\left(-\frac{(x - x_i(t))^2}{2\sigma^2}\right),
$$

and the flow field follows from the hydrodynamic identity $q_\ell = \rho_\ell \, u_\ell$. The bandwidth $\sigma = 10$ m provides sufficient smoothing to yield well-defined macroscopic fields while preserving spatial gradients of stop-and-go wave structures.

In regions of low density ($\rho \to 0$), the ratio $q/\rho$ becomes numerically ill-conditioned. To maintain gradient stability, a sigmoid-gated soft conditional is employed:

$$
u_\text{safe}(x, t) = \sigma_\beta(\rho - \varepsilon) \cdot \frac{q}{\rho + \varepsilon},
$$

where $\sigma_\beta(\cdot)$ denotes a logistic function with steepness $\beta = 20$. This formulation smoothly interpolates between zero (in empty regions) and the physical speed $q/\rho$ (in occupied regions), ensuring that gradients remain well-defined everywhere on the spatial grid.

### 3.2 Cross-lane aggregation

Real-world inductive loop detectors typically report aggregate measurements across all lanes at a given cross-section. To reflect this realistic constraint, per-lane fields are combined into cross-lane aggregates:

$$
q_\text{tot}(x, t) = \sum_\ell q_\ell(x, t), \qquad \rho_\text{tot}(x, t) = \sum_\ell \rho_\ell(x, t),
$$

$$
u_\text{tot}(x, t) = \frac{\sum_\ell q_\ell(x, t) \, u_\ell(x, t)}{q_\text{tot}(x, t) + \varepsilon}.
$$

The aggregated speed is flow-weighted, consistent with the definition of space-mean speed measured by dual-loop detectors. This aggregation step introduces an intentional information bottleneck: the lane-level structure of traffic states is discarded, and the reconstruction must proceed from the reduced cross-sectional signals alone.

### 3.3 Detector windowed extraction

At each detector location $x_d$, the aggregated macroscopic fields are spatially averaged over a window $[x_d - \Delta x, x_d + \Delta x]$:

$$
\hat{q}_d(t) = \frac{1}{|\mathcal{W}_d|} \sum_{m \in \mathcal{W}_d} q_\text{tot}(x_m, t), \qquad \hat{u}_d(t) = \frac{1}{|\mathcal{W}_d|} \sum_{m \in \mathcal{W}_d} u_\text{tot}(x_m, t),
$$

where $\mathcal{W}_d = \{m : |x_m - x_d| \leq \Delta x\}$ is the set of grid points within the detector window ($\Delta x = 10$ m). Detectors are spaced at regular intervals along the road segment (50 m in the baseline configuration), and the outputs are sampled at discrete observation times $t_k = k \, \Delta t_\text{obs}$ for $k = 1, \ldots, K$. The resulting observation tensor has shape $[K, N_d]$, where $K$ is the number of observation epochs and $N_d$ is the number of detector stations.


## 4. Loss Function and Optimization

### 4.1 Detector reconstruction loss

The primary objective function measures the discrepancy between simulated and observed detector outputs. For the predicted quantities $\hat{q}_d(t_k)$, $\hat{u}_d(t_k)$ and their observed counterparts $q_d^\text{obs}(t_k)$, $u_d^\text{obs}(t_k)$, the loss is defined as

$$
\mathcal{L}_\text{det} = w_q \, \bar{\mathcal{H}}(\hat{q}, q^\text{obs}) + w_u \, \bar{\mathcal{H}}(\hat{u}, u^\text{obs}),
$$

where $\bar{\mathcal{H}}$ denotes the element-wise Huber loss averaged over all valid observation pairs:

$$
\bar{\mathcal{H}}(y, y') = \frac{1}{|\mathcal{M}|} \sum_{(k,d) \in \mathcal{M}} \mathcal{H}_\delta(y_{k,d} - y'_{k,d}),
$$

with the validity mask $\mathcal{M} = \{(k,d) : y_{k,d} \text{ and } y'_{k,d} \text{ are both finite}\}$ and the Huber function

$$
\mathcal{H}_\delta(r) = \begin{cases} \frac{1}{2} r^2 & \text{if } |r| \leq \delta, \\ \delta \left(|r| - \frac{1}{2}\delta\right) & \text{otherwise.} \end{cases}
$$

The threshold $\delta = 1.0$ is used throughout. The channel weights are set to $w_q = 1.0$ for flow and $w_u = 0.1$ for speed. The Huber loss is preferred over mean squared error for its robustness to outlier observations arising from boundary effects or transient phenomena outside the IDM's representational capacity: the quadratic regime provides efficient learning for small residuals, while the linear tail prevents large errors from dominating the gradient signal.

### 4.2 Graph Laplacian regularization

An optional spatial smoothness prior can be imposed on the per-vehicle headway parameters through a graph Laplacian penalty defined over the leader-follower topology:

$$
\mathcal{R}(T) = \sum_{\substack{i=1 \\ \text{leader}(i) \neq \emptyset}}^{N} w_i \, (T_i - T_{\text{leader}(i)})^2,
$$

where the summation runs over all vehicles that have a valid same-lane leader. The edge weights $w_i$ may be set uniformly ($w_i = 1$) or computed as proximity-based platooning weights:

$$
w_i = \exp\!\left(-\frac{|s_i|}{s_\text{ref}}\right) \exp\!\left(-\frac{|\Delta v_i|}{v_\text{ref}}\right),
$$

which assign higher weight to closely-spaced vehicles traveling at similar speeds. When active, this regularizer is weighted by a coefficient $\lambda_T$ in the total loss.

### 4.3 Total loss

The total objective function used in the final reconstruction configuration is

$$
\mathcal{L} = \mathcal{L}_\text{det}.
$$

The graph Laplacian regularization ($\lambda_T = 0$) and the arrival-spread loss (discussed below) are excluded from the final configuration based on the ablation results presented in the experimental evaluation.

### 4.4 Arrival-spread loss (evaluated and excluded)

A second-order macroscopic constraint was investigated to incorporate wave propagation information into the loss function. For each detector $d$, a soft wave arrival time is computed by applying a sigmoid threshold to the observed speed drop below a baseline:

$$
\mu_d = \sum_k t_k \, P_{k,d}, \qquad P_{k,d} = \frac{\sigma_\tau(D_{k,d} - d_\text{thresh})}{\sum_{k'} \sigma_\tau(D_{k',d} - d_\text{thresh}) + \varepsilon},
$$

where $D_{k,d} = \bar{u}_d - u_d(t_k)$ is the speed deficit relative to the pre-wave baseline $\bar{u}_d$, and $\sigma_\tau$ denotes a logistic function with temperature $\tau$. The arrival-spread loss then penalizes the mismatch in temporal dispersion:

$$
\mathcal{L}_\text{arr} = \left(\text{std}(\{\mu_d^\text{sim}\}) - \text{std}(\{\mu_d^\text{obs}\})\right)^2.
$$

While this formulation is differentiable and theoretically captures wave propagation structure, systematic experiments showed that including $\mathcal{L}_\text{arr}$ consistently degraded trajectory reconstruction quality, increasing the macroscopic speed field RMSE by 0.02--0.04 m/s across all tested temporal resolutions. The likely mechanism is a conflict between the arrival-spread gradient and the detector reconstruction gradient at the current model fidelity, where the constant-ghost boundary condition limits the accuracy of wave timing in the simulated trajectories. This loss component is therefore excluded from the final framework.
