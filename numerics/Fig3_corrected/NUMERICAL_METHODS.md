# Numerical methods for the corrected many-body figure

## Model and realizations

We consider $N$ overdamped Brownian particles in a periodic square of side $L=5$, in units $k_B T=1$:

$$
d\mathbf x_a=\mu_a\left[-\nabla_a U+q_a E(t)\hat{\mathbf x}\right]dt+\sqrt{2\mu_a}\,d\mathbf W_a.
$$

For each realization, particle diameters $\sigma_a$ and charges $q_a$ are drawn independently and uniformly from $\{0.01,0.02,0.03\}$ and $\{0.9,1.0,1.1\}$, respectively. The mobilities and diffusion coefficients are $\mu_a=D_a=0.01/\sigma_a$. These particle parameters remain fixed throughout a trajectory and across field strengths within the same realization. Five realizations use independent disorder and Brownian-noise seeds, which are recorded with the data.

The pair potential uses minimum-image distances $r$ and $\sigma_{ab}=(\sigma_a+\sigma_b)/2$:

$$
U_{ab}(r)=U_{ab}^{\mathrm{WCA}}(r)+U^{\mathrm{s}}(r),
\qquad
U_{ab}^{\mathrm{WCA}}(r)=
\begin{cases}
4[(\sigma_{ab}/r)^{12}-(\sigma_{ab}/r)^6]+1,&r<2^{1/6}\sigma_{ab},\\
0,&r\geq2^{1/6}\sigma_{ab},
\end{cases}
$$

$$
U^{\mathrm{s}}(r)=
\begin{cases}
\dfrac{k}{2}[(r-r_0)^2-(l_0-r_0)^2],&r<l_0,\\
0,&r\geq l_0,
\end{cases}
\qquad k=5,\quad r_0=10^{-3},\quad l_0=2.
$$

The SM specifies the spring branch below its cutoff. We take the spring force to vanish above the cutoff and use the continuous shifted energy shown here, which preserves the specified force below it. The field is $E(t)=E[1+\cos(2\pi t/T)]$, with $T=1$. This is the original sinusoidal protocol with its time origin shifted to a field maximum. The calculations cover $N\in\{1,2,5,10,20,50,60,80,100\}$ and $E\in\{0.1,0.2,0.4,0.5,0.8,1.0\}$.

## Exact collective motion and relative-coordinate simulation

Define $\Gamma=\sum_a\mu_a^{-1}$, $Q_{\mathrm{ch}}=\sum_a q_a$, the unwrapped center of friction $\mathbf X=\Gamma^{-1}\sum_a\mathbf x_a/\mu_a$, and $\mathbf y_a=\mathbf x_a-\mathbf X$. Translation invariance gives the exact independent collective process

$$
d\mathbf X=\frac{Q_{\mathrm{ch}}}{\Gamma}E(t)\hat{\mathbf x}\,dt+\sqrt{\frac{2}{\Gamma}}\,d\mathbf W_{\mathrm{cm}}.
$$

The relative mobility is $M_{ab}=\mu_a\delta_{ab}-1/\Gamma$. The C++ code simulates the relative coordinates and integrates the collective contribution analytically. This removes collective Brownian noise from the numerical evaluation without changing the density observables.

Relative motion is integrated with projected, driven Metropolis-adjusted Langevin updates. A move associated with particle $a$ has direction $v_a=e_a-(\mu_a\Gamma)^{-1}\mathbf 1$ in particle-index space. Its scalar proposal has covariance $2\mu_a\Delta t$ per spatial direction and a tamed drift based on the force

$$
\mathbf F_a^{\mathrm{int}}+c_a E(t+\Delta t/2)\hat{\mathbf x},
\qquad c_a=q_a-\frac{Q_{\mathrm{ch}}}{\mu_a\Gamma}.
$$

For an increment $\boldsymbol\delta$, acceptance uses the logarithmic ratio

$$
-\Delta U+c_a E(t+\Delta t/2)\delta_x
+\log\frac{g(\mathbf y\mid\mathbf y')}{g(\mathbf y'\mid\mathbf y)},
$$

where $g$ is the actual Gaussian proposal density, including its tamed drift. Thus each update satisfies local detailed balance with the external work included. Forward and backward particle scans are chosen with equal probability. Recentring after a sweep is equivalent to projecting every accepted move because the interactions depend only on relative positions. The identity $\sum_a\mu_a v_a v_a^{\mathsf T}=M$ gives the required continuous-time relative generator as $\Delta t\to0$.

The production step is $\Delta t=2\times10^{-4}$. Separate calculations at smaller steps assess discretization error. Ten initial periods are discarded. Production lengths range from 160 to 10,000 periods, with longer records used for the low-field internal-work average. Exact lengths, steps and seeds are specified by the run metadata. Relative configurations are saved at every field extremum. Correlation analysis uses the final 160 periods, while the entropy-production average uses the complete production record. The single-particle case is evaluated directly from its exact wrapped Gaussian propagator.

## Entropy production

The true entropy production per period is the mean external work. The collective contribution is exact:

$$
\sigma_{\mathrm{cm}}=\frac{Q_{\mathrm{ch}}^2}{\Gamma}\int_0^T E(t)^2dt
=\frac{3T E^2Q_{\mathrm{ch}}^2}{2\Gamma}.
$$

Let $\Pi(t)=\sum_a q_a y_{a,x}(t)$. The remaining contribution is $W_{\mathrm{rel}}=\int_0^T E(t)\,d\Pi(t)$. It is recorded both as a displacement-work sum and through integration by parts:

$$
W_{\mathrm{rel}}=2E[\Pi(T)-\Pi(0)]-\int_0^T E'(t)\Pi(t)dt.
$$

The full boundary term is retained. The reported estimate is $\sigma_{\mathrm{true}}=\sigma_{\mathrm{cm}}+\langle W_{\mathrm{rel}}\rangle$. The two work calculations provide a quadrature check, and period blocks provide a sampling-error estimate. Sampled values are not clipped to theoretical limits. Exact comparison bounds are

$$
\sigma_{\mathrm{cm}}\leq\sigma_{\mathrm{true}}
\leq\frac{3TE^2}{2}\sum_a\mu_a q_a^2.
$$

## Density correlations and phase choice

For a $K\times K$ grid of square boxes of width $h=L/K$, the observable is the normalized particle count $\rho_m=n_m/N$. At lag $\tau=\ell T$, define $C_{nm}(\tau)=\langle\rho_n(t)\rho_m(t+\tau)\rangle$. Initial times are field maxima, field minima, or an equal mixture of these phases.

For either extremum, $E(t+s)=E(t+\tau-s)$ throughout an integer-period interval. The forward and reversed protocols therefore coincide, and the periodic initial and final distributions agree. This is the condition that permits the transposed-correlation entropy bound. Periodicity alone does not imply equality of forward and backward propagators. Pooling the two extremum phases also preserves the bound by joint convexity.

Uniform spatial translation makes $C_{nm}$ depend only on the box-index displacement $d=m-n$ modulo $K$. Define $P_d=K^2 C_{0d}$ and $P_d^{\mathrm R}=P_{-d}$, so that $\sum_d P_d=1$. The density bound per period is

$$
\sigma_{\mathrm{dens}}(K,\ell)=\frac{T}{\tau}\sum_d P_d\log\frac{P_d}{P_d^{\mathrm R}}.
$$

The candidate grids are $K\in\{5,10,20,40,80,160\}$ and the lag candidates are $\ell=1,\ldots,40$. Figure 3 uses $K=20$. Each grid uses its own lag and phase selected on the training data. Figure S3 compares $K=5,10,160$ on the same realizations.

## Positive evaluation of the box probabilities

At lag $\tau$, the collective displacement is Gaussian with mean $(Q_{\mathrm{ch}}E\tau/\Gamma,0)$ and variance $s^2=2\tau/\Gamma$ in each coordinate. Let $H(u_x,u_y)$ be the normalized distribution of relative pair displacements $\mathbf y_b(t+\tau)-\mathbf y_a(t)$ modulo $L$, averaged over all $N^2$ ordered particle pairs and the chosen initial phase. Same-particle pairs are included.

Writing $g_L(z;s)$ for the normalized wrapped one-dimensional Gaussian of variance $s^2$, define the nonnegative bin kernel

$$
\kappa_h(z;s)=\frac{1}{h}\int_{-h}^{h}(h-|u|)g_L(z+u;s)\,du.
$$

It includes both the uniform position within the initial box and the integration over the final box. The normalized displacement probabilities follow from

$$
P_{d_x,d_y}=\int H(\mathbf u)
\kappa_h(d_xh-u_x-Q_{\mathrm{ch}}E\tau/\Gamma;s)
\kappa_h(d_yh-u_y;s)\,d\mathbf u.
$$

The collective Gaussian and bin integrals are evaluated analytically. Relative pair displacements are accumulated with positive bilinear, or cloud-in-cell, interpolation on an auxiliary periodic $S\times S$ grid. Production choices are $S=128$ for $N\leq5$, $S=256$ for $5<N\leq20$, and $S=512$ for larger systems. This auxiliary grid is a numerical quadrature grid, distinct from the observation grid $K$. Grid refinement and direct Fourier comparisons quantify its error in the validation tables. Transverse reflection symmetry is averaged explicitly. Probabilities remain nonnegative, so the evaluation does not require a Fourier-tail clipping prescription.

## Independent training and evaluation

For each evaluated realization, the other four realizations provide the training relative-displacement histograms. These training histograms are combined with the evaluated realization's fixed $\Gamma$ and $Q_{\mathrm{ch}}$ to construct trial probabilities. The evaluated Brownian trajectory is excluded from all choices of lag, phase and fitted function.

The fitted function is $f_d=\operatorname{clip}[\log(P_d^{\mathrm{train}}/P_{-d}^{\mathrm{train}}),-c,c]$. Lag, phase and cap are selected jointly by cross-validation within the four training realizations. Each training realization is held out in turn while the function is fitted to the other three. The selected candidate maximizes the mean of these four training-validation scores. The cap candidates are $c\in\{8,12,16,20,24,28,32,40,50,64,80,100,150\}$. The final reported score on the independent evaluation record is

$$
\widehat\sigma_{\mathrm{est}}=\frac{T}{\tau}\left[
\sum_d\widehat P_d f_d-\sum_d\widehat P_d^{\mathrm R}(e^{f_d}-1)
\right].
$$

For any fixed $f$, its population value is at most $(T/\tau)D(P\Vert P^{\mathrm R})$. The score is linear in the evaluation probabilities, avoiding the upward bias of evaluating a fitted logarithmic ratio on its training sample. The training realizations may have different quenched particle parameters because the variational inequality holds for every fitted function. No choice is made using the evaluated trajectory's entropy production.

Independent training makes the statistical score an estimate of a lower bound in expectation, subject to the stated numerical quadrature and time-discretization errors. An individual finite-data value is not a certified lower confidence bound. The figure summarizes five evaluated realizations. Their standard deviation describes variation across particle realizations and trajectories; the cross-fitted estimates share training records and this spread is not an independent-sample standard error. Individual quality factors divide each estimated bound by the corresponding true entropy production. Figure 3(b) uses the ratio of the mean bound to the mean true entropy production, following the original plotting convention.

## Validation and reproducibility

The validation tables report single-particle analytic checks, equilibrium controls, time-step comparisons, auxiliary-grid refinement, work-quadrature agreement and the exact entropy-production comparison bounds. The output retains individual-run results, selected observation settings, training-cap comparisons, density-probability arrays, particle assignments and seeds. The supplied C++ and Python scripts regenerate the trajectories, analysis tables and vector figure from those recorded settings.
