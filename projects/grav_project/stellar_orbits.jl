### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ d44629e2-07a2-11f1-862d-ffeb4714407a
using Plots, PlutoUI, LinearAlgebra, LaTeXStrings, Measures, CurveFit, Statistics, ProgressLogging, PyFormattedStrings

# ╔═╡ f6f23886-7df7-41ea-878c-131198ba9f0e
md"""
# Stellar Orbits and Numerical Integration

### AS5523: Gravitational Dynamics and Accretion Physics
### By: Colton Quirk

The project description says to write the code in three parts:
1. The driver that inputs the initial position and velocity of the star.
2. The integrator that advances the star by one timestep
3. A function that calculates the acceleration at a given point due to a gravitational potential.

## 1. Acceleration due to a Potential

For the first part of the project we will be looking at orbits of stars around a point mass.
The acceleration due to this point mass is given by:

```math
\vec{a} = \frac{d\vec{v}}{dt} = \frac{\vec{F}}{m}=-\frac{M_{\rm ptmass}}{r^3} \vec{r}.
```

Given this we can write a function for this acceleration.
"""

# ╔═╡ 5a030e89-08a3-4ec3-958e-74a084a96a32
# a(pos, vel, M) = - M * pos / norm(pos)^3;

# ╔═╡ 63f4cb2d-768b-4154-8cfc-5a174046ce1c
G = M = 1.0

# ╔═╡ a33fbec9-4b51-47ca-872c-3c70e0986b20
function f(t, state)
	pos, vel = state
	acc = - G * M * pos / norm(pos)^3
	return [vel, acc]
end

# ╔═╡ 9b364800-93e9-48fe-a0cd-efcbd9c3f398
function energy(state)
	E = 0.5 * norm(state[2])^2 - G * M /norm(state[1])
end

# ╔═╡ cb529e73-d589-4b5e-8799-a01a9fa60f23
function momentum(state)
	L = cross(state[1], state[2])
end

# ╔═╡ 9760e4cd-5790-4843-b7b5-ba2d0190fa37
mutable struct Star{T}
	state::Array{T}
	pos_hist
	vel_hist
	e_hist
	l_hist
end

# ╔═╡ 5ccef611-c8bf-4124-820d-206e571f24fb
md"""
## 2. Integrator

Integrators I would like to try:

1. Eulerian
2. Symplectic Eulerian
3. Can't remember the name but my favorite one where you take the average of the two. This is related to the one in the project document.
   1. Leapfrog and Verlet Integration!!! (Yoshida integration?)
   2. Velocity-Verlet integration!!
5. 2nd Order Runge Kutta
6. 4th Order Runge Kutta
"""

# ╔═╡ 9703f6fb-dab5-46aa-af14-6600184b2a1f
md"""
Should also test the official integrators that come from the
> DifferentialEquations.jl
package.
"""

# ╔═╡ 53229c8f-3f5c-4195-9717-195a536c1dc6
md"""
I have set up my integrators to use the dynamic form of ODEs as discussed in Landau, Paez, Bordeianu (2024) Chapter 8.

```math
\frac{d \mathbf{S}(t)}{dt} = \mathbf{F}(t, \mathbf{S}).
```

Where the state of the system be defined as 

```math
\mathbf{S}=
\begin{bmatrix}
S^{(0)}(t) \\
S^{(1)}(t) \\
\ddots \\
S^{(N-1)}(t) \\
\end{bmatrix}, \qquad
\mathbf{F}=
\begin{bmatrix}
F^{(0)}(t, \mathbf{S}) \\
F^{(1)}(t, \mathbf{S}) \\
\ddots \\
F^{(N-1)}(t, \mathbf{S}) \\
\end{bmatrix}.
```
"""

# ╔═╡ cab6acf0-ddbf-4df9-8a7b-9af7a46159bc
md"""
That is to say that the state of the system is defined by an $N$-dimensional vector and the force that acts on the system is also an $N$ dimensional vector. My current issue in understanding is trying to combine this with the vector version of the force equation that I think is fantastic. There is no need to worry about keeping track of $x, y, z, v_x, v_y, v_z$ etc. Just plug in the position and velcities as vectors and out comes the force as a vector. 
I will attempt to make this work by setting up the following matrices to hold the state and force vectors:

```math
\mathbf{S}=
\begin{bmatrix}
x & y & z \\
v_x & v_y & v_z \\
\end{bmatrix},
\qquad
\mathbf{F}=
\begin{bmatrix}
v_x & v_y & v_z \\
a_x & a_y & a_z \\
\end{bmatrix}.
```
"""

# ╔═╡ daa960d0-b282-467f-8a93-43ed71a5b84f
md"""
The state $\mathbf{S}$ of the system where ``\mathbf{S}[0]`` (to use some python notation) is the position of the star, and ``\mathbf{S}[1]`` is the velocity of the star.
Therefore the force function ``\mathbf{F}`` will take in the state of the system, the current velocity becomes the force that is added to the position, the acceleration is calculated from the position/any other state variables, and then is added onto the velocity.

I will need to be careful about implementing the velocity-verlet method with this given the need to track the old and new positions. But that has to do more with the integrator being setup correctly than the force function.
"""

# ╔═╡ 1eed3115-ba8f-4a68-ab2b-9a8e66139d81
md"""
Okay now that the force equation is setup what does each integrator need? Ideally the function call of each integrator would have the same inputs. They should take in:

- The time, ``t``, of the simulation step
- The current state of the system $\mathbf{S}$
- The force function that determines the force experienced by the system ``\mathbf{F}``.
- The time step, ``\Delta t``, of the integration.

Is that all they neeed?
"""

# ╔═╡ d8803f0c-670e-4a00-8ed3-45ae8441e797
md"""
## Eulerian Integration

```math
\vec{x}_{n+1}=\vec{x}_n+\Delta t \times \vec{v}_n
```

```math
\vec{v}_{n+1}=\vec{v}_n+\Delta t \times \frac{d \vec{v}}{dt}
```

Probably should not use a symplectic integrator along the lines of:

```math
\vec{v}_{n+1}=\vec{v}_n+\Delta t \times \frac{d \vec{v}}{dt}
```

```math
\vec{x}_{n+1}=\vec{x}_n+\Delta t \times \vec{v}_{n+1}
```

At least for now.
"""

# ╔═╡ 11dd1844-e15a-4043-a342-449c098abf2b
function euler(t, y, force, dt)
	"""
	Function to calculate the next timestep using
		eulerian integration
	"""
	fR = force(t, y)
	y_new = y + fR * dt
	return y_new
end

# ╔═╡ 94b0f311-5784-4e2e-b2c6-36cc3f163207
function euler_symplectic(t, y, force, dt)
	"""
	Function to calculate the next timestep using
		symplectic eulerian integration
	"""
	y_new = copy(y)
	y_new[2] += force(t, y)[2] * dt
	y_new[1] += y_new[2] * dt
	return y_new
end

# ╔═╡ 805fe0ac-2acb-44d6-8ed5-b5d3543a8f03
function euler_2nd_order(t, y, force, dt)
	"""
	Function to calculate the next timestep using
		a second order euler-like scheme
	"""
	fR = force(t, y)

    y_half = y + 0.5 * fR * dt
    fR_half = force(t+0.5*dt, y_half)
    
    y_new = y + 0.5 * (fR + fR_half) * dt
	return y_new
end

# ╔═╡ 122adeac-702e-4dc8-8098-d216168984ca
function leapfrog(t, y, force, dt)
	fR = force(t, y)

	pos, vel = y

	# Kick
    vel_half = vel + 0.5 * fR[2] * dt

	# Drift
    pos_new = pos + vel_half * dt

	# Kick
    fR_new = force(t, [pos_new, vel_half])
    vel_new = vel_half + 0.5 * fR_new[2] * dt

    y_new = [pos_new, vel_new]
	return y_new
end

# ╔═╡ 4d60b037-1a21-49bf-aae7-8e79d08ae623
function velocity_verlet(t, y, force, dt)
	"""
	Function to calculate the next timestep using
		velocity verlet method
	"""
	pos, vel = y
	fR = force(t, y)
	new_pos = pos + vel * dt + 0.5 * fR[2] * (dt^2)

	fR_next = force(t, [new_pos, vel])
	new_vel = vel + 0.5 * (fR[2] + fR_next[2]) * dt

	y_new = [new_pos, new_vel]
	return y_new
end

# ╔═╡ 49b29633-eca2-46ee-aa39-842a4504759c
function rk2(t, y, force, dt)
	"""
	Function to calculate the next timestep using
		2nd order Runge-Kutta
	"""
	k1 = dt * force(t, y)
    k2 = dt * force(t + 0.5 * dt, y + 0.5 * k1)
    
    y_new = y + k2
end

# ╔═╡ b3f79f29-3488-45d2-b839-4dcfe20e1eaf
md"""
This is really interesting, `rk2` seems to increase the energy over time, while `rk4` seems to derease the energy over time. 
At least in the example wih ``\Delta t = 5\times 10^{-1}`` and ``t_{end} = 50*6*\pi``
Is this actually the case?
I know that the Runge-Kutta methods do not have energy conservation built in but do the different orders have different biases for what happens with the energy?
"""

# ╔═╡ cb26d768-2a0f-4581-96e1-205b98bd4981
function rk4(t, y, force, dt)
	"""
	Function to calculate the next timestep using
		4th-order Runge-Kutta
	"""
	k1 = force(t, y)
    k2 = force(t + 0.5 * dt, y + 0.5 * k1 * dt)
    k3 = force(t + 0.5 * dt, y + 0.5 * k2 * dt)
    k4 = force(t + dt, y + k3 * dt)
    
    y_new = y + dt * (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0
end

# ╔═╡ 4bde3946-fdf8-4bf2-8888-92eef8eda718
function rk4_bonnell(t, y, force, dt)
    pos, vel = y
	
    hdt = dt * 0.5

    # k1
    _, accel0 = force(t, [pos, vel])
    vel1 = vel + accel0*hdt
    pos1 = pos + vel1*hdt

    # k2
    _, accel1 = force(t + hdt, [pos1, vel1])
    vel2 = vel + accel1*hdt
    pos2 = pos + vel2*hdt

    # k3
    _, accel2 = force(t + hdt, [pos2, vel2])
    vel3 = vel + accel2*hdt
    pos3 = pos + vel3*hdt

    # k4
    _, accel3 = force(t + dt, [pos3, vel3])

    # final update
    vel_new = vel + dt*(accel0 + 2.0*accel1 + 2.0*accel2 + accel3)/6.0
	# Needto use the updated vel
	# previously had vel instead of vel_new which did not work
    pos_new = pos + dt*(vel_new + 2.0*vel1 + 2.0*vel2 + vel3)/6.0

    return [pos_new, vel_new]
end

# ╔═╡ 135ee28d-74cf-41b0-b41c-8d15f712af16
function yoshida(t, y, force, dt)
	"""
	Function to calculate the next timestep using
		4th-order Yoshida integration
	"""
	w0 = - cbrt(2.0)/ (2.0 - cbrt(2.0))
    w1 = 1.0 / (2.0 - cbrt(2.0))
    c1 = c4 = w1/2.0
    c2 = c3 = 0.5 * (w0 + w1)
    d1 = d3 = w1
    d2 = w0

    pos = y[1]
    vel = y[2]

    # I feel like there should be a more elegant solution for this
    pos1 = pos + c1 * vel * dt
    # Currently the force function is independent of time so it doesn't matter that I don't update time
    # But shouldn't the change in t be added to the t in the function?
    vel1 = vel + d1 * force(t, [pos1, vel])[2] * dt

    pos2 = pos1 + c2 * vel1 * dt
    vel2 = vel1 + d2 * force(t, [pos2, vel1])[2] * dt

    pos3 = pos2 + c3 * vel2 * dt
    vel3 = vel2 + d3 * force(t, [pos3, vel2])[2] * dt

    pos4 = pos3 + c4 * vel3 * dt
    vel4 = vel3

    y_new = [pos4, vel4]

    return y_new
end

# ╔═╡ 45ceae60-6686-47f4-9f38-fc0d349921a2
md"""
The `Yoshida` integrator is really interesting.
It has very good properties that seem to lend itself to this problem.
However it is limited in that the problem needs to be time reversible, given that some of the timesteps are backwards in time.
"""

# ╔═╡ ac439b9a-7b25-45e5-8c70-624ab88b6359
all_integrators = [euler, euler_symplectic, euler_2nd_order, leapfrog, velocity_verlet, rk2, rk4, rk4_bonnell, yoshida]

# ╔═╡ cbbf7c1f-f0b7-4422-8516-1ded7c6a596f
integrator_labels = [euler euler_symplectic euler_2nd_order leapfrog velocity_verlet rk2 rk4 rk4_bonnell yoshida]

# ╔═╡ 6bba2ae7-889d-41a1-a414-e5800361a573
md"""
## 3. Simulation/Driver

Now we can use the integrator and the force calculation from above to simulate the orbit of the star around the point mass.
"""

# ╔═╡ 14778872-d513-44c2-9379-0e7d1c7a5fd6
md"""
!!! info "Adaptive Timestep"
	> You will need to use a timestep control such as dt << sqrt(r/a)\
"""

# ╔═╡ 80b53ae2-0e59-4b4f-9c4b-13de29eaed5d
function orbit(t_start, t_end, dt, star, force, integrator, calc_energy)
	t = t_start
	# t_vals = [t]
	while t < t_end
		new_state = integrator(t, star.state, force, dt)
		star.state = new_state
		push!(star.pos_hist, star.state[1])
		push!(star.vel_hist, star.state[2])
		push!(star.e_hist, calc_energy(star.state))
		push!(star.l_hist, momentum(star.state))
		t += dt
		# push!(t_vals, t)
	end
	star.pos_hist = reduce(hcat, star.pos_hist)'
	star.vel_hist = reduce(hcat, star.vel_hist)'
	star.l_hist = reduce(hcat, star.l_hist)'
end

# ╔═╡ 1d55f0e2-adc1-4a7c-aa6c-9fbd721d117b
function orbit_adaptive(t_start, t_end, dt_init, star, force, integrator, calc_energy; η=0.05)
	t = t_start
	dt = dt_init

	t_vals = [t_start]

	while t < t_end
		x = star.state[1]
		r = norm(x)

		a_vec = force(t, star.state)[2]
		a = norm(a_vec)

		# Adaptive timestep
		max_step = η * sqrt(r / a) # correct units

		# Clamp timestep
		dt = minimum([dt_init, max_step])

		# Step forward
		new_state = integrator(t, star.state, force, dt)
		star.state = new_state

		# Record position, energy, angular momentum
		push!(star.pos_hist, star.state[1])
		push!(star.vel_hist, star.state[2])
		push!(star.e_hist, calc_energy(star.state))
		push!(star.l_hist, momentum(star.state))

		# Update time
		t += dt
		push!(t_vals, t)
	end
	
	star.pos_hist = reduce(hcat, star.pos_hist)'
	star.vel_hist = reduce(hcat, star.vel_hist)'
	star.l_hist = reduce(hcat, star.l_hist)'
	return t_vals
end

# ╔═╡ 5f798ff3-f463-4bbe-9c31-b08ffb9d6d37
md"""
Start ``x``: $( @bind x0 Slider(0.5:1e-1:2.0, default=1.0, show_value=true) ) \
\
Start ``v_{y}``: $( @bind vy0 Slider(0.5:1e-1:2.0, default=1.0, show_value=true) ) \
\
``\Delta``t: $( @bind Δt Slider(1e-4:1e-3:1., default=1e-2, show_value=true) ) \
\
``n_{\mathrm{orbits}}``: $( @bind norbits Slider(1:500, default=3, show_value=true) ) \
\
Adaptive timestep integrator: 
$( @bind adap_int Select([euler, euler_symplectic, euler_2nd_order, leapfrog,
								velocity_verlet, rk2, rk4, yoshida], default=yoshida) ) \
\
Select Integrators: \
$( @bind integrators MultiCheckBox(all_integrators) )
"""

# ╔═╡ 5fd4d32b-995e-458a-9a7f-c12de9a6327f
begin
	start_pos = [x0, 0.0, 0.0]
	start_vel = [0.0, vy0, 0.0]
	
	start_state = [start_pos,
				  start_vel]
end

# ╔═╡ 505a6e03-e123-4253-b719-971022018d3b
begin
	r = norm(start_pos)
	a_vec = f(0.0, start_state)[2]
	a = norm(a_vec)
	minimum_dt_circular = 0.05 * sqrt(r / a)
end;

# ╔═╡ f9e281cf-0c30-40a6-9d06-e257f2d78df5
md"""
Adaptive timestep Δt for a circular orbit: $minimum_dt_circular
"""

# ╔═╡ 6c06982d-ac87-45b0-a885-92efdf325aea
t_end = 2π * norbits;

# ╔═╡ bc8fa8f7-8c66-4108-9a74-f1587d944c99
begin
	stars = []
	for integrator in integrators
		star = Star(start_state,
				    [start_state[1]],
					[start_state[2]],
				    [energy(start_state)],
				    [momentum(start_state)])
		orbit(0.0, t_end, Δt, star, f, integrator, energy)
		push!(stars, star)
	end
end

# ╔═╡ 05496e2c-04ae-4467-bb50-7f692968f3fb
begin
	star_adaptive = Star(start_state,
					    [start_state[1]],
						[start_state[2]],
					    [energy(start_state)],
					    [momentum(start_state)])
	adap_tvals = orbit_adaptive(0.0, t_end, Δt, star_adaptive, f, adap_int, energy)
end;

# ╔═╡ 51cb4863-772e-4784-830d-cccbb37e891f
t_vals = 0.0:Δt:t_end+Δt;

# ╔═╡ 1d21d0fd-6e47-4913-b05f-662b2d04d727
begin
	θ = 0:1e-2:2π
	p = plot(cos.(θ), sin.(θ),
		 label="Idealized Orbit", aspect_ratio=1,
		 legend=:outertopright)

	plot!(star_adaptive.pos_hist[:, 1], star_adaptive.pos_hist[:, 2], 
		  label="adaptive")

	for (i, star) in enumerate(stars)
		label = integrators[i]
		plot!(star.pos_hist[:, 1], star.pos_hist[:, 2], 
			  label=label)
	end

	plot!(title="Stellar Orbits around a Point Mass " * L"(\Delta t="*"$Δt)")
	plot!(xlabel=L"x")
	plot!(ylabel=L"y")
	plot!(size=(800, 600))
	lim = 1.5
	plot!(xlim=(-lim, lim))
	plot!(ylim=(-lim, lim))
	# plot(p, p_e, p_l, layout=(3, 1))
end

# ╔═╡ 5b4ec8f0-07d6-45b9-a20c-30b52271f359
begin
	p_e = plot(title="Relative Change in Energy over Time", 
			   xlabel=L"t", ylabel=L"\Delta E / E_{0}", 
			   legend=:outertopright,
			  )
	
	for (i, star) in enumerate(stars)
		label = integrators[i]
		plot!(t_vals, (star.e_hist .- first(star.e_hist)) ./ abs(first(star.e_hist)), label=label)
	end
	
	plot!(adap_tvals, (star_adaptive.e_hist .- first(star_adaptive.e_hist)) ./ abs(first(star_adaptive.e_hist)), label="adaptive")
	plot!(size=(1200, 600))
	plot!(margin=10mm)
	# plot!(yscale=:log10)
	# p_e
end

# ╔═╡ 723a4485-c64e-48fa-907e-3c5ea0316023
begin
	p_l = plot(title="Relative Change in the Angular Momentum over Time",
			   xlabel=L"t", ylabel=L"\Delta L / L_{0}", 
			   legend=:outertopright)
	for (i, star) in enumerate(stars)
		label = integrators[i]
		l_hist = map(norm, eachrow(star.l_hist))
		plot!(t_vals[1:end], (l_hist .- first(l_hist)) ./ abs(first(l_hist)), label=label)
	end

	adap_l_hist = map(norm, eachrow(star_adaptive.l_hist))
	plot!(adap_tvals[1:end], (adap_l_hist .- first(adap_l_hist)) ./ abs(first(adap_l_hist)), label="adaptive")
	plot!(size=(1200, 600))
	plot!(margin=10mm)
	p_l
end

# ╔═╡ 3287fb59-f3ce-4763-8715-7457418ef7be
md"""
## 3. 1 Numerical Accuracy
"""

# ╔═╡ 739fc6c8-3305-4715-bc45-7741489cb63e
md"""
!!! info "Numerical Accuracy"
	> Once the integrator works reasonably well and has been tested, **calculate how the error in the integration depends on the timestep used**.\
	> **Evaluate at what timestep the integration breaks completely on one orbit.**\
	> Errors can be estimate by keeping track of quantities that should be conserved such as total energy and angular momentum.\
	> For a simple Keplerian orbit, one can also used the semi-major axis or apastron, although measuring these can be difficult.
"""

# ╔═╡ 144cff2f-bf84-460d-b17c-433fc3bfd614
md"""
Check this for `euler`, `2nd_order_euler`, `rk2`, and `rk4`

Symplectic integrators will be harder to deal with I think. Just have to use the maximum error in the energy rather than the final error.
"""

# ╔═╡ 2951df38-84e4-45a9-b75c-d8f56b87d23b
init_energy = energy(start_state)

# ╔═╡ 8317da7d-d6b1-4d4a-82cf-389f85eef719
init_momentum = momentum(start_state)

# ╔═╡ a4078281-1160-4455-a9fc-97df9c2b8062
md"""
### 3.1.1 Error Dependence on Timestep
"""

# ╔═╡ debe9624-fb48-4633-a94f-b457040eb6c0
begin
	dt_vals = exp10.(range(-4, stop=log10(0.5), length=100))
	# dt_vals = range(1e-4, 1e-1, 100)
end

# ╔═╡ 5774aba4-8294-46fb-9623-c0c165f08301
dt_vals[end]

# ╔═╡ ca182723-485a-4664-b2ee-d054b10379a6
begin
	all_change_in_E = Vector{Float64}[]

	@progress for integrator in all_integrators
		int_errors = []
		@progress for Δt in dt_vals
			star = Star(start_state,
				    [start_state[1]],
					[start_state[2]],
				    [energy(start_state)],
				    [momentum(start_state)])
			orbit(0.0, 2π, Δt, star, f, integrator, energy)
			# ΔE = (star.e_hist[end] - star.e_hist[1]) / abs(star.e_hist[1])
			# Need to take the maximum of the error asthe symplectic integrators oscillate
			# between energy values so the final energy might not be a good indicator of the true error
			ΔE = maximum(abs.(star.e_hist .- star.e_hist[1]))
			push!(int_errors, ΔE)
		end
		push!(all_change_in_E, int_errors)
	end
end

# ╔═╡ 10f46658-3ac6-49c6-9820-d7a56ef6db6c
all_change_in_E

# ╔═╡ 6f729716-7ac9-427d-9df0-9f869947f322
begin
	scatter(log10.(dt_vals), [log10.(E_err) for E_err in all_change_in_E], label=integrator_labels, legend=:outertopright, size=(1600, 700),
		xlabel=L"\log{(\Delta t)}", ylabel=L"\log{ (| \Delta E |) }", margin=10mm, title="Relative change in energy per timestep",
		)
end

# ╔═╡ 64c2ed54-2ef5-4b4f-8464-497f1a8ffce1
begin
	prob = CurveFitProblem(log10.(dt_vals), log10.(abs.(all_change_in_E[1])))
	sol = solve(prob, LinearCurveFitAlgorithm())
	
	sol.u
end

# ╔═╡ 1749ea8b-1c36-48b2-841d-d00f8e482d6f
slopes = [0.58, 2.0, 0.64, 3.96, 3.95, 2.94, 5.2, 7.51]

# ╔═╡ b12a629d-fe8c-426a-8f0a-83fa93055935
begin
	err_p = plot(title="Log-Log Error Scaling", xlabel=L"\log(\Delta t)", ylabel=L"\log(|\Delta E|)",
         legend=:outertopright, size=(1600, 700), margin=10mm)

	end_cutoff = 45
	start_cutoffs = [1, 1, 1, 23, 23, 1, 40, 1, 65]

	for (i, int_errors) in enumerate(all_change_in_E)
	    abs_errors = abs.(int_errors)
	    valid = abs_errors .> 0

		if (i == 1) || (i == 3)
			log_dt  = log10.(dt_vals[valid])[start_cutoffs[i]:end_cutoff]
	   		log_err = log10.(abs_errors[valid])[start_cutoffs[i]:end_cutoff]
		else
			log_dt  = log10.(dt_vals[valid])[start_cutoffs[i]:end]
	   		log_err = log10.(abs_errors[valid])[start_cutoffs[i]:end]
		end
	
	    
	    n = cov(log_dt, log_err) / var(log_dt)
	
	    # Measured points
	    scatter!(err_p, log_dt, log_err, label="$(all_integrators[i]) (data)", color=i)
	
	    # Best-fit line
	    C = mean(log_err) - n * mean(log_dt)
	    plot!(err_p, log_dt, n .* log_dt .+ C,
	          label="$(all_integrators[i]) (slope=$(round(n,digits=2)))",
	          linestyle=:dash, color=i)
	end
	
	# display(err_p)
	err_p
end

# ╔═╡ 3b19c567-c772-47b8-a671-62420f48800e
md"""
### 3.1.2 Timestep to "break" one orbit
"""

# ╔═╡ b8fd5f9e-d2b5-40f0-9025-d0db531164c3
md"""
This part of the instructions does not make much sense. \
What does it mean for an orbit to "breaks completely"? \
My interpretation of this was to set some tolerance value such that if the orbit ends outside of that tolerance than it has broken.
"""

# ╔═╡ a2ba4ab8-d599-4785-87bd-b38c9337f370
tol = 1e-4

# ╔═╡ 9e7fb844-d442-4ec9-bbbe-84960c5c4130
dt_vals

# ╔═╡ 5f4869d4-f3dd-49fc-8e5f-56e1ab0be2e7
begin
	break_dt_vals = []
	for integrator in all_integrators
		for Δt in dt_vals
			star = Star(start_state,
				    [start_state[1]],
					[start_state[2]],
				    [energy(start_state)],
				    [momentum(start_state)])
			orbit(0.0, 2π, Δt, star, f, integrator, energy)
			err = norm(star.state[1] - start_state[1])
			if err > tol
				push!(break_dt_vals, Δt)
				break
			end
		end
	end
	break_dt_vals
end

# ╔═╡ 639d9fea-848b-4212-9df7-8a98fa2e2a04
2π/1e-7

# ╔═╡ 3e1f5517-1cb1-4d25-afc6-ade63999e48f
md"""
# 4. Orbits in Galactic Potentials
"""

# ╔═╡ 928a29e7-af2a-419e-b2b4-7650fa3fb41e
md"""
```math
\phi = v_{0}^{2} \ln \left( R_{c}^{2} + x^{2} + \frac{y^2}{q_{1}^{2}} + \frac{z^{2}}{q_{2}^2} \right)
```
"""

# ╔═╡ 37ed9b7a-3294-4b05-969e-052e83997e7c
md"""
```math
m\frac{d \vec{v}}{dt}=
-\nabla \phi = 
-\frac{2v_0^2}{R_c^2+x^2+y^2/q_1^2+z^2/q_2^2}
\left[
x\ \hat{x} + \frac{1}{q_1^2}y\ \hat{y} + \frac{1}{q_2^2}z\ \hat{z}
\right]
```
"""

# ╔═╡ 818719aa-b636-4db4-859c-3bbd30e23317
function gal_energy_general(state, potential)
	E = 0.5 * norm(state[2])^2 + potential(state)
end

# ╔═╡ f92b206c-786d-4e75-b402-80217f33ddca
function gal_general(t, state; v_0=1.0, R_c=0.2, q_1=1.0, q_2=1.0)
	pos, vel = state
	x, y, z = pos
	axis = [1.0, 1/(q_1^2), 1/(q_2^2)]
	
    c = -2*v_0^2/(R_c^2 + x^2 + y^2/(q_1^2) + z^2/(q_2^2))

	acc = c * pos .* axis

	fR = [vel, acc]
	return fR
end

# ╔═╡ dd2b7266-db26-4605-b1da-69b91adf7856
60π / 1e-3

# ╔═╡ bf57d43a-aede-43ac-9572-55630bbae190

@bind gal_params confirm(
	PlutoUI.combine() do Child
		md"""
		**Initial Position:**\
		``x_{0}``: $( Child( Slider(0.0:0.01:5.0, default=0.2, show_value=true) ) ) \
		``y_{0}``: $( Child( Slider(0.0:0.01:5.0, default=0.0, show_value=true) ) ) \
		``z_{0}``: $( Child( Slider(0.0:0.01:5.0, default=0.0, show_value=true) ) ) \
		\
		**Initial Velocity:**\
		``V_{x,0}``: $( Child( Slider(-3.0:0.1:3.0, default=0.0, show_value=true) ) ) \
		``V_{y,0}``: $( Child( Slider(-3.0:0.01:3.0, default=1.0, show_value=true) ) ) \
		``V_{z,0}``: $( Child( Slider(-3.0:0.1:3.0, default=0.1, show_value=true) ) ) \
		\
		**Galacitc Potential Conditions**\
		``v_{0}`` = $( Child( Slider(0.1:0.01:1.0, default=1.0, show_value=true) ) ) \
		``R_{c}`` = $( Child( Slider(0.1:0.01:1.0, default=0.2, show_value=true) ) ) \
		``q_{1}``: $( Child( Slider(0.1:0.01:1.0, default=1.0, show_value=true) ) ) \
		``q_{2}``: $( Child( Slider(0.1:0.01:1.0, default=1.0, show_value=true) ) ) \
		\
		**Integration**\
		Galaxy integrator: $( Child( Select([rk4, rk4_bonnell, yoshida], default = yoshida) ) ) \
		\
		"""
	end
)


# ╔═╡ aa4b97e0-377e-42b5-9d5b-27e2b0027143
x_0, y_0, z_0, vx_0, vy_0, vz_0, v_0, R_c, q_1, q_2, gal_int = gal_params;

# ╔═╡ 97fb16d6-c3f2-461b-8ffe-478674913d86
function Φ_L_general(state; q_1=1.0, q_2=1.0)
	pos, vel = state
	x, y, z = pos
	PE = v_0^2 * log(R_c^2 + x^2 + y^2/q_1^2 + z^2/q_2^2)
	return PE
end

# ╔═╡ fb04e075-a96b-4365-b7aa-ee0a8db7285c
gal(t, state) = gal_general(t, state, v_0=v_0, R_c = R_c, q_1=q_1, q_2=q_2)

# ╔═╡ 0f8e6aa3-acf6-4882-a0d7-e742a221acea
potential(state) = Φ_L_general(state, q_1=q_1, q_2=q_2)

# ╔═╡ c16d355c-0e47-4c1d-9f48-894e3817422c
gal_energy(state) = gal_energy_general(state, potential)

# ╔═╡ d036ef8e-36d4-4445-9af5-4c8a719cbe64
begin
	pos_0 = [x_0, y_0, z_0]
	vel_0 = [vx_0, vy_0, vz_0]
	gal_start = [pos_0, vel_0]
	
	p_gal = gal_start[1]
	r_gal = norm(p_gal)
	v_gal = gal_start[2]
	vrot = norm(cross(p_gal, v_gal)) / r_gal
end;

# ╔═╡ bb72b814-6f07-4109-8f61-9cfe833d834e
begin
	star_gal = Star(gal_start,
				    [gal_start[1]],
					[gal_start[2]],
				    [gal_energy(gal_start)],
				    [momentum(gal_start)])
	dt_gal = 1e-3
	tvals_gal = orbit_adaptive(0.0, 60π, dt_gal, star_gal, gal, gal_int, gal_energy; η=1e-3) # , dt_min=dt_gal/64.0)
end;

# ╔═╡ 1521b9aa-40e8-4687-a865-e733a9c71d6e
md"""
**Plotting**\
3D: $( @bind gal_3d Switch(default=false)) \
just xy: $( @bind just_xy Switch(default=true))
"""

# ╔═╡ 47ae380b-3329-4b24-a9e2-288ec1c512aa
function plot_star(star)
	xy = plot(star_gal.pos_hist[:, 1], star_gal.pos_hist[:, 2];
			xlabel='x', ylabel='y', label="", )# aspect_ratio=1)
	plot!(xy, size=(800, 800))
	
	plot!(plot_title="Stellar Orbit in Galactic Potential")
	# plot!(title="Stellar Orbits around a Point Mass $Δt")
	# plot!()
	plot!(margin=10mm)
	# lim = 1.5
	# plot!(xlim=(-lim, lim))
	# plot!(ylim=(-lim, lim))
end

# ╔═╡ 71c01fb0-80b3-4e2f-a527-316fd992e2d5
begin
	if gal_3d
		plot(star_gal.pos_hist[:, 1], star_gal.pos_hist[:, 2], star_gal.pos_hist[:, 3];
			xlabel='x', ylabel='y', zlabel='z', label="")
		plot!(size=(1600, 700))
	elseif just_xy 
		xy = plot(star_gal.pos_hist[:, 1], star_gal.pos_hist[:, 2];
				xlabel='x', ylabel='y', label="", )# aspect_ratio=1)
		plot!(size=(800, 800))
	else
		xy = plot(star_gal.pos_hist[:, 1], star_gal.pos_hist[:, 2]; # star_gal.pos_hist[:, 3], 
				xlabel='x', ylabel='y', label="") # , aspect_ratio=:equal)
	
		xz = plot(star_gal.pos_hist[:, 1], star_gal.pos_hist[:, 3]; 
				  xlabel='x', ylabel='z', label="") # , aspect_ratio=:equal)
	
		yz = plot(star_gal.pos_hist[:, 2], star_gal.pos_hist[:, 3];
				 xlabel='y', ylabel='z', label="") # , aspect_ratio=:equal)
		plot(xy, xz, yz, layout=(1, 3))
		plot!(size=(1600, 700))
	end
	
	plot!(plot_title="Stellar Orbit in Galactic Potential")
	# plot!(title="Stellar Orbits around a Point Mass $Δt")
	# plot!()
	plot!(margin=10mm)
	# lim = 1.5
	# plot!(xlim=(-lim, lim))
	# plot!(ylim=(-lim, lim))
end

# ╔═╡ a276362c-4c11-42bb-bfd7-02f52bd68bb0
begin
	e = plot(tvals_gal, star_gal.e_hist, label=L"E", ylabel="Energy", legend=:outertopright)
	l = plot(tvals_gal, star_gal.l_hist, label=[L"L_{x}" L"L_{y}" L"L_{z}"], ylabel="Angular Momentum", legend=:outertopright)
	t = plot(tvals_gal[1:end-1], diff(tvals_gal), label=L"\Delta t", ylabel=L"\Delta t", xlabel="Time", legend=:outertopright)
	plot(e, l, t, layout=(3, 1), plot_title="Energy and Angular Momentum History")
	plot!(size=(1600, 700))
	plot!(left_margin=10mm, bottom_margin=10mm)
end

# ╔═╡ e8a7cf79-3e5d-4f0a-98cc-af7eba1bdbd2
md"""
## 4.1 Various Galactic Orbits
"""

# ╔═╡ 16f7c3f2-d836-41da-bac8-ac0ba1b6fdfb
function classify_orbit(star; tol=1e-10)
	# Set floating point errors to 0.0
	signs = sign.(ifelse.(abs.(star.l_hist) .< tol, 0.0, star.l_hist))
	# If the sign remains constant throughout 
	# the orbit than it is a loop orbit
	# if the sign changes than it is a box orbit
	if map(allequal, eachcol(signs))[3]
		return :loop
	else
		return :box
	end
end

# ╔═╡ bc0e23f9-a5a1-4848-bbc2-ab467d1e4f09
@bind survey_params confirm(
	PlutoUI.combine() do Child
		md"""
		``n_{rows}`` = $( Child( Slider(2:1:6, default=2.0, show_value=true) ) ) \
		**Galacitc Potential Conditions**\
		``v_{0}`` = $( Child( Slider(0.1:0.01:1.0, default=1.0, show_value=true) ) ) \
		``R_{c}`` = $( Child( Slider(0.1:0.01:1.0, default=0.2, show_value=true) ) ) \
		``q_{1}``: $( Child( Slider(0.1:0.01:1.0, default=1.0, show_value=true) ) ) \
		``q_{2}``: $( Child( Slider(0.1:0.01:1.0, default=1.0, show_value=true) ) ) \
		\
		"""
	end
)

# ╔═╡ 5ed243c0-104b-4f68-bcba-348fe5deba97
n_rows_survey, v_0_survey, r_c_survey, q_1_survey, q_2_survey = survey_params

# ╔═╡ 1f6073ab-5c5f-4caa-b8a2-f568f2c2dbcd
gal_survey(t, state) = gal_general(t, state; v_0=v_0_survey, R_c=r_c_survey, q_1=q_1_survey, q_2=q_2_survey)

# ╔═╡ f6b6010e-032f-49ed-bde8-11f1df819885
survey_potential(state) = Φ_L_general(state, q_1=q_1_survey, q_2=q_2_survey)

# ╔═╡ d7642bbd-1937-47ed-9686-591ded4ad0f8
survey_energy(state) = gal_energy_general(state, survey_potential)

# ╔═╡ 87a8be99-30eb-4246-8bc7-0e0a5f6b6fd9
begin
	xs  = range(0.01, 3.0, length=n_rows_survey)      # initial x values
	vys = range(0.01, 3.0, length=n_rows_survey)      # initial vy values

	box_color = "#0072B2"
	loop_color  = "#D55E00"
	
	plt = plot(layout = (n_rows_survey, n_rows_survey), size=(1600,1600),
			  plot_title = "Orbit survey in triaxial potential: "*L"q_1 = %$q_1_survey,\; q_2 = %$q_2_survey"*" (Orange=Loop, Blue=Box)",
			  plot_titlefontsize = 24,
			  titlefontsize=18)
	
	@progress for (i, vy0) in enumerate(vys)      # rows
	    for (j, x0) in enumerate(xs)    # columns

			row_from_top = n_rows_survey + 1 - i
			idx = (row_from_top - 1)*n_rows_survey + j
	        # idx = (i-1)*5 + j          # subplot index 1..25

			ic = [[x0, 0.0, 0.0],
				  [0.0, vy0, 0.0]]
			star = Star(ic, [ic[1]], [ic[2]], [survey_energy(ic)], [momentum(ic)])
	
	        orbit_adaptive(0.0, 30*2π, 1e-3, star, gal_survey, yoshida, survey_energy)  # longer integration = better 

			type = classify_orbit(star)

			if type == :loop
				c = loop_color
				# style = :solid
			elseif type == :box
				c = box_color
				# style = :dash
			else 
				c = "green"
				# style=:solid
			end
	
	        plot!(plt[idx], star.pos_hist[:, 1], star.pos_hist[:, 2];
	              xlabel = "",
	              ylabel = "",
	              title = L"x=%$(round(x0,digits=2)),\; v_{y}=%$(round(vy0,digits=2))",
				  color = c,
				  # linestyle=style,
	              legend = false,
				 framestyle=:none)
	              # aspect_ratio = :equal)
	    end
	end
	plt
end

# ╔═╡ f048c295-56a9-47c6-9198-2ce09b3d6982


# ╔═╡ f21752a2-afdb-4c4c-b8b6-faeb396a23de
md"""
## 4. 2 ``(r, v_{\mathrm{rot}})`` Plane
Need to determine where box and loop orbits are in the ``(r, v_{\mathrm{rot}})`` plane.
Where:
```math
v_{\mathrm{rot}}=\frac{\vec{v} \times \vec{r}}{r}
```
"""

# ╔═╡ d021c01d-9a91-4900-b2e7-f94442870766
begin
	plot(star_gal.pos_hist[:, 1], star_gal.vel_hist[:, 1])
	scatter!([R_c], [v_0])
end

# ╔═╡ d8067735-c715-4f65-932f-7d6d8ed49a51
begin
	scatter([R_c], [1.0], label="Circular Orbit for:\n" * L"q_1=1.0" * ", " * L"q_2=1.0")
	scatter!([r_gal], [vrot], label="star")
	scatter!([0.05], [range(0.1, 3.0, length=100)[end-6]], label="sample")
	plot!(xlabel=L"r")
	plot!(ylabel=L"v_{rot}")
	plot!(title="Orbits in the " * L"(r, v_{rot})" * " plane")
	plot!(xlim=(0.0, 3.0))
	plot!(ylim=(0, 3.0))
	plot!(size=(600, 600))
	plot!(aspect=1)
	plot!(legend=:outertopright)
end

# ╔═╡ 83910760-e461-43be-87d7-7e8ebb154731
md"""
>Such orbits have no particular sense of circulation about the center and thus their time-averaged angular momentum is zero\
> - Binney and Tremaine (2008)
"""

# ╔═╡ af2d6c23-211c-4669-bacd-5f2267b01e38
map(mean, eachcol(star_gal.l_hist))

# ╔═╡ 8eb2bea0-dfc9-4be1-80b0-cd14fa587213
classify_orbit(star_gal)

# ╔═╡ f8d84352-9bc2-4fe9-81c9-411fa0c3d42c
@bind plot_r_vrot Switch()

# ╔═╡ 46270c42-11a9-4c0b-a397-1baf4ba73c71
begin
	if plot_r_vrot
		npixs = 100
		r_vals = range(0.01, 3.0, length=npixs)
		v_rot_vals = range(0.01, 3.0, length=npixs)
	
		orbit_types = Matrix{Symbol}(undef, npixs, npixs)
	
		@progress for (i, r) in enumerate(r_vals)
		    for (j, v_rot) in enumerate(v_rot_vals)
		        ic = [[r, 0.0, 0.0],
					  [0.0, v_rot, 0.0]]
		        star = Star(ic, [ic[1]], [ic[2]], [energy(ic)], [momentum(ic)])
		        orbit_adaptive(0.0, 10*2π, dt_gal, star, gal_survey, yoshida, survey_energy)  # longer integration = better classification
		        orbit_types[i, j] = classify_orbit(star)
		    end
		end
		
		# Convert to numeric for heatmap: 1=loop, 0=box
		numeric = Float64.(orbit_types .== :loop)
		
		heatmap(r_vals, v_rot_vals, permutedims(numeric),
		        xlabel="r", ylabel="v_rot", 
		        title="Orbit structure: yellow=loop, blue=box",
		        colorbar=false, color=:viridis,
			   aspect=1, size=(800, 800))
		savefig(hm, "report/images/r_vrot_plane.png")
	end
end

# ╔═╡ 4afbadf8-e387-46a0-9903-e077f9f7b5f6
md"""
## 4.3 Surface-of-Section Plots
"""

# ╔═╡ 2e86c3ab-b3e7-4faa-afa7-1edd7a26296c
function vel_for_E(state, E = -0.337)
	vx = sqrt(2(E-Φ_L(state)))
	return vx
end

# ╔═╡ d03f4fc4-ccff-4d61-8146-5fe890037cee
begin
	E = -0.337
	ic = [[0.5, 0.0, 0.0],
		  [0.0, 0.0, 0.0]]
	vx = sqrt(2(E-survey_potential(ic)))
end

# ╔═╡ ac7cf451-9b31-4f4b-9c1a-1c398e42a0a8


# ╔═╡ 325a6250-5b68-46af-9ad7-08a4645af429
zs = star_gal.pos_hist[:, 3][1:end]

# ╔═╡ d19f75d6-daa3-4ace-961e-aa42c239a464
mapslices(norm, star_gal.pos_hist, dims=2)

# ╔═╡ 3b1819df-0fab-4e8b-8974-72409e287588
function v_rot(star)
	returnvrot = []
	Rs = mapslices(norm, star.pos_hist, dims=2)
	for (i, pos) in enumerate(eachrow(star.pos_hist))
		vel = star.vel_hist[i, :]
		push!(returnvrot, norm(cross(vel, pos))/Rs[i])
	end
	return returnvrot
end

# ╔═╡ e943a7bc-ac2f-48e3-9970-bb84880ede93
function surface_of_section(star)
	# Find points where the orbit passes from z - to +
	zs = star.pos_hist[:, 3]
	shiftzs = circshift(zs, -1)
	indx = (zs[1:end] .< 0.).*(shiftzs[1:end] .> 0.)

	Rs = mapslices(norm, star.pos_hist, dims=2)
	vRs = v_rot(star)
    return (Rs[1:end][indx], vRs[1:end][indx])
end

# ╔═╡ c26ae70a-55b3-4928-b270-ef128c8c77eb
plot_Rs, plot_vrot = surface_of_section(star_gal)

# ╔═╡ b3806c7c-e849-4386-92d8-295fdea4aa44
scatter(plot_Rs, plot_vrot)

# ╔═╡ ffb93cdf-d4dd-4c66-b88d-bad81d748681


# ╔═╡ 991a397f-e72b-4271-a182-4b0085aa667a


# ╔═╡ ff5ed656-0d26-42a5-bff4-8d6b709eb8e6
md"""
# 5. Animation
"""

# ╔═╡ 04ac60fd-ec95-49cd-aea4-2125f8722dc6
size(tvals_gal)

# ╔═╡ a186a66d-d99a-49db-bfd7-24c18513eeec
@bind calc_frames Slider(100:100:100000, default=1000, show_value=true)

# ╔═╡ 8ac244cb-3f7c-45bc-9021-d535dc00fed8
nframes = size(tvals_gal[1:calc_frames:end])

# ╔═╡ 0c2af742-fd02-4e44-bed4-e1373facd206
md"""
Seconds in video: $( nframes[1]/30 )
"""

# ╔═╡ c5293299-c0a4-4233-aeca-6d6267bd6516
@bind skip confirm(Slider(100:100:100000, default=1000, show_value=true))

# ╔═╡ 4ac02b34-4f0f-47c5-be2a-3646c7347290
@bind anim_or_not Switch()

# ╔═╡ dc130e84-4719-4aa7-894a-bb020a1224ae
function animate_star(star, fname; skip=1000, fps=10)
    anim_hist = star.pos_hist[1:skip:end, :]
    n = size(anim_hist, 1)
    x, y, z = anim_hist[:,1], anim_hist[:,2], anim_hist[:,3]
    
    xlims = (minimum(x)-0.1, maximum(x)+0.1)
    ylims = (minimum(y)-0.1, maximum(y)+0.1)
    zlims = (minimum(z)-0.1, maximum(z)+0.1)

    anim = @animate for i in 1:n
        xy = scatter([x[i]], [y[i]]; xlabel="x", ylabel="y", xlims=xlims, ylims=ylims, label="")
        plot!(xy, x[1:i], y[1:i], label="")

		plot(xy, size=(800,800), margin=10mm)

        # xz = scatter([x[i]], [z[i]]; xlabel="x", ylabel="z", xlims=xlims, ylims=zlims, label="")
        # plot!(xz, x[1:i], z[1:i], label="")

        # yz = scatter([y[i]], [z[i]]; xlabel="y", ylabel="z", xlims=ylims, ylims=zlims, label="")
        # plot!(yz, y[1:i], z[1:i], label="")

        # plot(xy, xz, yz, layout=(1,3), size=(1600,700), margin=10mm)
    end
    gif(anim, fname, fps=fps)
end

# ╔═╡ 1110f7f3-c508-4db1-b2fb-59db33548b01
f"anim_gal_{x_0:.2f}_{q_1:.2f}_{q_2:.2f}.gif"

# ╔═╡ 8c1f2c5a-a59f-431f-b1f8-fba0519c97f2
if anim_or_not
	fname = f"anim_gal_{x_0:.2f}_{q_1:.2f}_{q_2:.2f}.mp4"
	gif1 = animate_star(star_gal, fname; skip=skip, fps=30)
end

# ╔═╡ 8dea93f9-f3fc-4e6a-998c-0372c94fe640
function animate_star_3d(star; skip=1000, fps=20)
	anim_hist = star.pos_hist[1:skip:end, :]
	n = size(anim_hist, 1)
    x, y, z = anim_hist[:,1], anim_hist[:,2], anim_hist[:,3]
    
    xlims = (minimum(x)-0.1, maximum(x)+0.1)
    ylims = (minimum(y)-0.1, maximum(y)+0.1)
    zlims = (minimum(z)-0.1, maximum(z)+0.1)
	
	anim_3d = @animate for i in 1:n
		scatter([x[i]], [y[i]], [z[i]];
            xlabel="x", ylabel="y", zlabel="z", label="",
            xlims=xlims, ylims=ylims, zlims=zlims,
            plot_title="Stellar Orbit in Galactic Potential",
            size=(1600, 700), margin=10mm
        )
        plot!(x[1:i], y[1:i], z[1:i], label="")
	end

	gif(anim_3d, fps=fps)
end

# ╔═╡ 44c27737-0c9e-4cfd-866c-73d70ecb8356
# if anim_or_not
# 	animate_star_3d(star_gal; skip=skip, fps=60)
# end

# ╔═╡ bd810cdd-aa6f-4fe1-9427-ed574f9bf31d


# ╔═╡ 795e4e12-9f8a-4a3a-8554-cb297bdb60f1
html"""
<style>
	main {
		margin: 0 auto;
		max-width: 2000px;
    	padding-left: max(160px, 10%);
    	padding-right: max(160px, 10%);
	}
</style>
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CurveFit = "5a033b19-8c74-5913-a970-47c3779ef25c"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Measures = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
ProgressLogging = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
PyFormattedStrings = "5f89f4a4-a228-4886-b223-c468a82ed5b9"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
CurveFit = "~1.7.0"
LaTeXStrings = "~1.4.0"
Measures = "~0.3.3"
Plots = "~1.41.5"
PlutoUI = "~0.7.77"
ProgressLogging = "~0.1.6"
PyFormattedStrings = "~0.1.13"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.6"
manifest_format = "2.0"
project_hash = "731ddde5e6fae2cb504d9bf720e9cfa61d9af4b0"

[[deps.ADTypes]]
git-tree-sha1 = "f7304359109c768cf32dc5fa2d371565bb63b68a"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.21.0"

    [deps.ADTypes.extensions]
    ADTypesChainRulesCoreExt = "ChainRulesCore"
    ADTypesConstructionBaseExt = "ConstructionBase"
    ADTypesEnzymeCoreExt = "EnzymeCore"

    [deps.ADTypes.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "2eeb2c9bef11013efc6f8f97f32ee59b146b09fb"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.44"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "35ea197a51ce46fcd01c4a44befce0578a1aaeca"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.5.0"

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

    [deps.Adapt.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "78b3a7a536b4b0a747a0f296ea77091ca0a9f9a3"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.23.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceAMDGPUExt = "AMDGPU"
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = ["CUDSS", "CUDA"]
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceMetalExt = "Metal"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fde3bf89aead2e723284a8ff9cdf5b551ed700e8"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.5+0"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.CommonSolve]]
git-tree-sha1 = "78ea4ddbcf9c241827e7035c3a03e2e456711470"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.6"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConcreteStructs]]
git-tree-sha1 = "f749037478283d372048690eb3b5f92a79432b34"
uuid = "2569d6c7-a4a2-43d3-a901-331e8e4be471"
version = "0.2.3"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "d9d26935a0bcffc87d2613ce14c527c99fc543fd"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.0"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.CurveFit]]
deps = ["ADTypes", "CommonSolve", "ConcreteStructs", "DifferentiationInterface", "Distributions", "FastRationals", "ForwardDiff", "InverseFunctions", "LinearAlgebra", "LinearSolve", "Markdown", "NonlinearSolveBase", "NonlinearSolveFirstOrder", "PrecompileTools", "RecursiveArrayTools", "SciMLBase", "Setfield", "StatsAPI"]
git-tree-sha1 = "28dd4698d6c4cea689a27055b26ec81ae074c046"
uuid = "5a033b19-8c74-5913-a970-47c3779ef25c"
version = "1.7.0"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "e357641bb3e0638d353c4b29ea0e40ea644066a6"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.3"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.DifferentiationInterface]]
deps = ["ADTypes", "LinearAlgebra"]
git-tree-sha1 = "7ae99144ea44715402c6c882bfef2adbeadbc4ce"
uuid = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
version = "0.7.16"

    [deps.DifferentiationInterface.extensions]
    DifferentiationInterfaceChainRulesCoreExt = "ChainRulesCore"
    DifferentiationInterfaceDiffractorExt = "Diffractor"
    DifferentiationInterfaceEnzymeExt = ["EnzymeCore", "Enzyme"]
    DifferentiationInterfaceFastDifferentiationExt = "FastDifferentiation"
    DifferentiationInterfaceFiniteDiffExt = "FiniteDiff"
    DifferentiationInterfaceFiniteDifferencesExt = "FiniteDifferences"
    DifferentiationInterfaceForwardDiffExt = ["ForwardDiff", "DiffResults"]
    DifferentiationInterfaceGPUArraysCoreExt = "GPUArraysCore"
    DifferentiationInterfaceGTPSAExt = "GTPSA"
    DifferentiationInterfaceMooncakeExt = "Mooncake"
    DifferentiationInterfacePolyesterForwardDiffExt = ["PolyesterForwardDiff", "ForwardDiff", "DiffResults"]
    DifferentiationInterfaceReverseDiffExt = ["ReverseDiff", "DiffResults"]
    DifferentiationInterfaceSparseArraysExt = "SparseArrays"
    DifferentiationInterfaceSparseConnectivityTracerExt = "SparseConnectivityTracer"
    DifferentiationInterfaceSparseMatrixColoringsExt = "SparseMatrixColorings"
    DifferentiationInterfaceStaticArraysExt = "StaticArrays"
    DifferentiationInterfaceSymbolicsExt = "Symbolics"
    DifferentiationInterfaceTrackerExt = "Tracker"
    DifferentiationInterfaceZygoteExt = ["Zygote", "ForwardDiff"]

    [deps.DifferentiationInterface.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DiffResults = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
    Diffractor = "9f5e2b26-1114-432f-b630-d3fe2085c51c"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FastDifferentiation = "eb9bf01b-bf85-4b60-bf87-ee5de06c00be"
    FiniteDiff = "6a86dc24-6348-571c-b903-95158fe2bd41"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    GTPSA = "b27dd330-f138-47c5-815b-40db9dd9b6e8"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    PolyesterForwardDiff = "98d1487c-24ca-40b6-b7ab-df2af84e126b"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
    SparseMatrixColorings = "0a514795-09f3-496d-8182-132a7b665d35"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "fbcc7610f6d8348428f722ecbe0e6cfe22e672c6"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.123"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.EnumX]]
git-tree-sha1 = "c49898e8438c828577f04b92fc9368c388ac783c"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.7"

[[deps.EnzymeCore]]
git-tree-sha1 = "24bbb6fc8fb87eb71c1f8d00184a60fc22c63903"
uuid = "f151be2c-9106-41f4-ab19-57ee4f262869"
version = "0.8.19"

    [deps.EnzymeCore.extensions]
    AdaptExt = "Adapt"
    EnzymeCoreChainRulesCoreExt = "ChainRulesCore"

    [deps.EnzymeCore.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "27af30de8b5445644e8ffe3bcb0d72049c089cf1"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.7.3+0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.ExproniconLite]]
git-tree-sha1 = "c13f0b150373771b0fdc1713c97860f8df12e6c2"
uuid = "55351af7-c7e9-48d6-89ff-24e801d99491"
version = "0.10.14"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "95ecf07c2eea562b5adbd0696af6db62c0f52560"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.5"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "01ba9d15e9eae375dc1eb9589df76b3572acd3f2"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "8.0.1+0"

[[deps.FastClosures]]
git-tree-sha1 = "acebe244d53ee1b461970f8910c235b259e772ef"
uuid = "9aa1b823-49e4-5ca5-8b0f-3971ec8bab6a"
version = "0.3.2"

[[deps.FastRationals]]
git-tree-sha1 = "441d4b21b2897653adfe3eb64939865c53cb2905"
uuid = "275e499e-7a09-5551-a1d1-9fe535a2b717"
version = "0.3.2"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "2f979084d1e13948a3352cf64a25df6bd3b4dca3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.16.0"

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStaticArraysExt = "StaticArrays"
    FillArraysStatisticsExt = "Statistics"

    [deps.FillArrays.weakdeps]
    PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Setfield"]
git-tree-sha1 = "9340ca07ca27093ff68418b7558ca37b05f8aeb1"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.29.0"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffSparseArraysExt = "SparseArrays"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cddeab6487248a39dae1a960fff0ac17b2a28888"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.3.3"

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

    [deps.ForwardDiff.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "2c5512e11c791d1baed2049c5652441b28fc6a31"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.FunctionWrappersWrappers]]
deps = ["FunctionWrappers", "PrecompileTools", "TruncatedStacktraces"]
git-tree-sha1 = "6874da243fb93e34201d7d4587ffa0e920682f64"
uuid = "77dc65aa-8811-40c2-897b-53d922fa7daf"
version = "1.0.0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "b7bfd56fa66616138dfe5237da4dc13bbd83c67f"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.1+0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "ee0585b62671ce88e48d3409733230b401c9775c"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.22"

    [deps.GR.extensions]
    IJuliaExt = "IJulia"

    [deps.GR.weakdeps]
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "7dd7173f7129a1b6f84e0f03e0890cd1189b0659"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.22+0"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "24f6def62397474a297bfcec22384101609142ed"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.86.3+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "5e6fe50ae7f23d171f44e311c2960294aaa0beb5"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.19"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "68c173f4f449de5b438ee67ed0c9c748dc31a2ec"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.28"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "ec1debd61c300961f98064cfb21287613ad7f303"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "b3ad4a0255688dcb895a52fafbaae3023b588a90"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.4.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.Jieko]]
deps = ["ExproniconLite"]
git-tree-sha1 = "2f05ed29618da60c06a87e9c033982d4f71d0b6c"
uuid = "ae98c720-c025-4a4a-838c-29b094483192"
version = "0.2.1"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6893345fd6658c8e475d40155789f4860ac3b21"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.4+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.Krylov]]
deps = ["LinearAlgebra", "Printf", "SparseArrays"]
git-tree-sha1 = "c4d19f51afc7ba2afbe32031b8f2d21b11c9e26e"
uuid = "ba0b0d4f-ebba-5204-a429-3ac8c609bfb7"
version = "0.10.6"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "44f93c47f9cd6c7e431f2f2091fcba8f01cd7e8f"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.10"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "97bbca976196f2a1eb9607131cb108c69ec3f8a6"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.41.3+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "f04133fe05eff1667d2054c53d59f9122383fe05"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.2+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d0205286d9eceadc518742860bf23f703779a3d6"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.41.3+0"

[[deps.LineSearch]]
deps = ["ADTypes", "CommonSolve", "ConcreteStructs", "FastClosures", "LinearAlgebra", "MaybeInplace", "PrecompileTools", "SciMLBase", "SciMLJacobianOperators", "StaticArraysCore"]
git-tree-sha1 = "9f7253c0574b4b585c8909232adb890930da980a"
uuid = "87fe0de2-c867-4266-b59a-2f0a94fc965b"
version = "0.1.6"

    [deps.LineSearch.extensions]
    LineSearchLineSearchesExt = "LineSearches"

    [deps.LineSearch.weakdeps]
    LineSearches = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LinearSolve]]
deps = ["ArrayInterface", "ConcreteStructs", "DocStringExtensions", "EnumX", "GPUArraysCore", "InteractiveUtils", "Krylov", "Libdl", "LinearAlgebra", "MKL_jll", "Markdown", "OpenBLAS_jll", "PrecompileTools", "Preferences", "RecursiveArrayTools", "Reexport", "SciMLBase", "SciMLLogging", "SciMLOperators", "Setfield", "StaticArraysCore"]
git-tree-sha1 = "1ddad5f2b0717f71f1588b3519e2f7d80fc2ce65"
uuid = "7ed4a6bd-45f5-4d41-b270-4a48e9bafcae"
version = "3.69.0"

    [deps.LinearSolve.extensions]
    LinearSolveAMDGPUExt = "AMDGPU"
    LinearSolveAlgebraicMultigridExt = "AlgebraicMultigrid"
    LinearSolveBLISExt = ["blis_jll", "LAPACK_jll"]
    LinearSolveBandedMatricesExt = "BandedMatrices"
    LinearSolveBlockDiagonalsExt = "BlockDiagonals"
    LinearSolveCUDAExt = "CUDA"
    LinearSolveCUDSSExt = "CUDSS"
    LinearSolveCUSOLVERRFExt = ["CUSOLVERRF", "SparseArrays"]
    LinearSolveChainRulesCoreExt = "ChainRulesCore"
    LinearSolveCliqueTreesExt = ["CliqueTrees", "SparseArrays"]
    LinearSolveEnzymeExt = ["EnzymeCore", "SparseArrays"]
    LinearSolveFastAlmostBandedMatricesExt = "FastAlmostBandedMatrices"
    LinearSolveFastLapackInterfaceExt = "FastLapackInterface"
    LinearSolveForwardDiffExt = "ForwardDiff"
    LinearSolveGinkgoExt = ["Ginkgo", "SparseArrays"]
    LinearSolveHYPREExt = "HYPRE"
    LinearSolveIterativeSolversExt = "IterativeSolvers"
    LinearSolveKernelAbstractionsExt = "KernelAbstractions"
    LinearSolveKrylovKitExt = "KrylovKit"
    LinearSolveMetalExt = "Metal"
    LinearSolveMooncakeExt = "Mooncake"
    LinearSolvePETScExt = ["PETSc", "SparseArrays"]
    LinearSolveParUExt = ["ParU_jll", "SparseArrays"]
    LinearSolvePardisoExt = ["Pardiso", "SparseArrays"]
    LinearSolveRecursiveFactorizationExt = "RecursiveFactorization"
    LinearSolveSparseArraysExt = "SparseArrays"
    LinearSolveSparspakExt = ["SparseArrays", "Sparspak"]

    [deps.LinearSolve.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    AlgebraicMultigrid = "2169fc97-5a83-5252-b627-83903c6c433c"
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockDiagonals = "0a1fb500-61f7-11e9-3c65-f5ef3456f9f0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    CUSOLVERRF = "a8cc9031-bad2-4722-94f5-40deabb4245c"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    CliqueTrees = "60701a23-6482-424a-84db-faee86b9b1f8"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FastAlmostBandedMatrices = "9d29842c-ecb8-4973-b1e9-a27b1157504e"
    FastLapackInterface = "29a986be-02c6-4525-aec4-84b980013641"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Ginkgo = "4c8bd3c9-ead9-4b5e-a625-08f1338ba0ec"
    HYPRE = "b5ffcf37-a2bd-41ab-a3da-4bd9bc8ad771"
    IterativeSolvers = "42fd0dbc-a981-5370-80f2-aaf504508153"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    KrylovKit = "0b1a1467-8014-51b9-945f-bf0ae24f4b77"
    LAPACK_jll = "51474c39-65e3-53ba-86ba-03b1b862ec14"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    PETSc = "ace2c81b-2b5f-4b1e-a30d-d662738edfe0"
    ParU_jll = "9e0b026c-e8ce-559c-a2c4-6a3d5c955bc9"
    Pardiso = "46dd5b70-b6fb-5a00-ae2d-e8fea33afaf2"
    RecursiveFactorization = "f2c3362d-daeb-58d1-803e-2bc74f2840b4"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Sparspak = "e56a9233-b9d6-4f03-8d0f-1825330902ac"
    blis_jll = "6136c539-28a5-5bf0-87cc-b183200dce32"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f00544d95982ea270145636c181ceda21c4e2575"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.2.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "282cadc186e7b2ae0eeadbd7a4dffed4196ae2aa"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.2.0+0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MaybeInplace]]
deps = ["ArrayInterface", "LinearAlgebra", "MacroTools"]
git-tree-sha1 = "54e2fdc38130c05b42be423e90da3bade29b74bd"
uuid = "bb5d69b7-63fc-4a16-80bd-7e42200c7bdb"
version = "0.1.4"
weakdeps = ["SparseArrays"]

    [deps.MaybeInplace.extensions]
    MaybeInplaceSparseArraysExt = "SparseArrays"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ff69a2b1330bcb730b9ac1ab7dd680176f5896b8"
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.1010+0"

[[deps.Measures]]
git-tree-sha1 = "b513cedd20d9c914783d8ad83d08120702bf2c77"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.Moshi]]
deps = ["ExproniconLite", "Jieko"]
git-tree-sha1 = "53f817d3e84537d84545e0ad749e483412dd6b2a"
uuid = "2e0e35c7-a2e4-4343-998d-7ef72827ed2d"
version = "0.3.7"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.NonlinearSolveBase]]
deps = ["ADTypes", "Adapt", "ArrayInterface", "CommonSolve", "Compat", "ConcreteStructs", "DifferentiationInterface", "EnzymeCore", "FastClosures", "LinearAlgebra", "LogExpFunctions", "Markdown", "MaybeInplace", "PreallocationTools", "PrecompileTools", "Preferences", "Printf", "RecursiveArrayTools", "SciMLBase", "SciMLJacobianOperators", "SciMLLogging", "SciMLOperators", "SciMLStructures", "Setfield", "StaticArraysCore", "SymbolicIndexingInterface", "TimerOutputs"]
git-tree-sha1 = "a89529d343dbb09670a24df090787dc3475fba5d"
uuid = "be0214bd-f91f-a760-ac4e-3421ce2b2da0"
version = "2.19.0"

    [deps.NonlinearSolveBase.extensions]
    NonlinearSolveBaseBandedMatricesExt = "BandedMatrices"
    NonlinearSolveBaseChainRulesCoreExt = "ChainRulesCore"
    NonlinearSolveBaseEnzymeExt = ["ChainRulesCore", "Enzyme"]
    NonlinearSolveBaseForwardDiffExt = "ForwardDiff"
    NonlinearSolveBaseLineSearchExt = "LineSearch"
    NonlinearSolveBaseLinearSolveExt = "LinearSolve"
    NonlinearSolveBaseMooncakeExt = "Mooncake"
    NonlinearSolveBaseReverseDiffExt = "ReverseDiff"
    NonlinearSolveBaseSparseArraysExt = "SparseArrays"
    NonlinearSolveBaseSparseMatrixColoringsExt = "SparseMatrixColorings"
    NonlinearSolveBaseTrackerExt = "Tracker"

    [deps.NonlinearSolveBase.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    LineSearch = "87fe0de2-c867-4266-b59a-2f0a94fc965b"
    LinearSolve = "7ed4a6bd-45f5-4d41-b270-4a48e9bafcae"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SparseMatrixColorings = "0a514795-09f3-496d-8182-132a7b665d35"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.NonlinearSolveFirstOrder]]
deps = ["ADTypes", "ArrayInterface", "CommonSolve", "ConcreteStructs", "FiniteDiff", "ForwardDiff", "LineSearch", "LinearAlgebra", "LinearSolve", "MaybeInplace", "NonlinearSolveBase", "PrecompileTools", "Reexport", "SciMLBase", "SciMLJacobianOperators", "Setfield", "StaticArraysCore"]
git-tree-sha1 = "eea7cbe389b168c77df7ff779fb7277019c685c8"
uuid = "5959db7a-ea39-4486-b5fe-2dd0bf03d60d"
version = "2.0.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "NetworkOptions", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "1d1aaa7d449b58415f97d2839c318b70ffb525a0"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.6.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e2bb57a313a74b8104064b7efd01406c0a50d2ff"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.6.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.44.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "e4cff168707d441cd6bf3ff7e4832bdf34278e4a"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.37"
weakdeps = ["StatsBase"]

    [deps.PDMats.extensions]
    StatsBaseExt = "StatsBase"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0662b083e11420952f2e62e17eddae7fc07d5997"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.57.0+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "db76b1ecd5e9715f3d043cec13b2ec93ce015d53"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.44.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "26ca162858917496748aad52bb5d3be4d26a228a"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.4"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "1cc8ad0762e59e713ee3ef28f9b78b2c9f4ca078"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.41.5"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "6ed167db158c7c1031abf3bd67f8e689c8bdf2b7"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.77"

[[deps.PreallocationTools]]
deps = ["Adapt", "ArrayInterface", "PrecompileTools"]
git-tree-sha1 = "e16b73bf892c55d16d53c9c0dbd0fb31cb7e25da"
uuid = "d236fae5-4411-538c-8e31-a6e3d9e00b46"
version = "1.2.0"

    [deps.PreallocationTools.extensions]
    PreallocationToolsForwardDiffExt = "ForwardDiff"
    PreallocationToolsReverseDiffExt = "ReverseDiff"
    PreallocationToolsSparseConnectivityTracerExt = "SparseConnectivityTracer"

    [deps.PreallocationTools.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "07a921781cab75691315adc645096ed5e370cb77"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.3"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "522f093a29b31a93e34eaea17ba055d850edea28"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "f0803bc1171e455a04124affa9c21bba5ac4db32"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.6"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.PyFormattedStrings]]
deps = ["PrecompileTools", "Printf"]
git-tree-sha1 = "4edb7868d6a1c9ef22c8132c4aa1857e56bd4a84"
uuid = "5f89f4a4-a228-4886-b223-c468a82ed5b9"
version = "0.1.13"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "34f7e5d2861083ec7596af8b8c092531facf2192"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.8.2+2"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll"]
git-tree-sha1 = "da7adf145cce0d44e892626e647f9dcbe9cb3e10"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.8.2+1"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "9eca9fc3fe515d619ce004c83c31ffd3f85c7ccf"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.8.2+1"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "8f528b0851b5b7025032818eb5abbeb8a736f853"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.8.2+2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9da16da70037ba9d701192e27befedefb91ec284"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.2"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "DocStringExtensions", "GPUArraysCore", "LinearAlgebra", "PrecompileTools", "RecipesBase", "StaticArraysCore", "SymbolicIndexingInterface"]
git-tree-sha1 = "e4fd3369c78666a65ccec25dba28a0b181434ab2"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "3.52.0"

    [deps.RecursiveArrayTools.extensions]
    RecursiveArrayToolsFastBroadcastExt = "FastBroadcast"
    RecursiveArrayToolsForwardDiffExt = "ForwardDiff"
    RecursiveArrayToolsKernelAbstractionsExt = "KernelAbstractions"
    RecursiveArrayToolsMeasurementsExt = "Measurements"
    RecursiveArrayToolsMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    RecursiveArrayToolsReverseDiffExt = ["ReverseDiff", "Zygote"]
    RecursiveArrayToolsSparseArraysExt = ["SparseArrays"]
    RecursiveArrayToolsStatisticsExt = "Statistics"
    RecursiveArrayToolsStructArraysExt = "StructArrays"
    RecursiveArrayToolsTablesExt = ["Tables"]
    RecursiveArrayToolsTrackerExt = "Tracker"
    RecursiveArrayToolsZygoteExt = "Zygote"

    [deps.RecursiveArrayTools.weakdeps]
    FastBroadcast = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "5b3d50eb374cea306873b371d3f8d3915a018f0b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.9.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.RuntimeGeneratedFunctions]]
deps = ["ExprTools", "SHA", "Serialization"]
git-tree-sha1 = "7257165d5477fd1025f7cb656019dcb6b0512c38"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.17"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SciMLBase]]
deps = ["ADTypes", "Accessors", "Adapt", "ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "EnumX", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "Moshi", "PreallocationTools", "PrecompileTools", "Preferences", "Printf", "RecipesBase", "RecursiveArrayTools", "Reexport", "RuntimeGeneratedFunctions", "SciMLLogging", "SciMLOperators", "SciMLPublic", "SciMLStructures", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface"]
git-tree-sha1 = "908c0bf271604d09393a21c142116ab26f66f67c"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "2.154.0"

    [deps.SciMLBase.extensions]
    SciMLBaseChainRulesCoreExt = "ChainRulesCore"
    SciMLBaseDifferentiationInterfaceExt = "DifferentiationInterface"
    SciMLBaseDistributionsExt = "Distributions"
    SciMLBaseEnzymeExt = "Enzyme"
    SciMLBaseForwardDiffExt = "ForwardDiff"
    SciMLBaseMLStyleExt = "MLStyle"
    SciMLBaseMakieExt = "Makie"
    SciMLBaseMeasurementsExt = "Measurements"
    SciMLBaseMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    SciMLBaseMooncakeExt = "Mooncake"
    SciMLBasePartialFunctionsExt = "PartialFunctions"
    SciMLBasePyCallExt = "PyCall"
    SciMLBasePythonCallExt = "PythonCall"
    SciMLBaseRCallExt = "RCall"
    SciMLBaseReverseDiffExt = "ReverseDiff"
    SciMLBaseTrackerExt = "Tracker"
    SciMLBaseZygoteExt = ["Zygote", "ChainRulesCore"]

    [deps.SciMLBase.weakdeps]
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DifferentiationInterface = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    MLStyle = "d8e11817-5142-5d16-987a-aa16d5891078"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    PartialFunctions = "570af359-4316-4cb7-8c74-252c00c2016b"
    PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
    PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
    RCall = "6f49c342-dc21-5d91-9882-a32aef131414"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.SciMLJacobianOperators]]
deps = ["ADTypes", "ArrayInterface", "ConcreteStructs", "ConstructionBase", "DifferentiationInterface", "FastClosures", "LinearAlgebra", "SciMLBase", "SciMLOperators"]
git-tree-sha1 = "e96d5e96debf7f80a50d0b976a13dea556ccfd3a"
uuid = "19f34311-ddf3-4b8b-af20-060888a46c0e"
version = "0.1.12"

[[deps.SciMLLogging]]
deps = ["Logging", "LoggingExtras", "Preferences"]
git-tree-sha1 = "0161be062570af4042cf6f69e3d5d0b0555b6927"
uuid = "a6db7da4-7206-11f0-1eab-35f2a5dbe1d1"
version = "1.9.1"

    [deps.SciMLLogging.extensions]
    SciMLLoggingTracyExt = "Tracy"

    [deps.SciMLLogging.weakdeps]
    Tracy = "e689c965-62c8-4b79-b2c5-8359227902fd"

[[deps.SciMLOperators]]
deps = ["Accessors", "ArrayInterface", "DocStringExtensions", "LinearAlgebra"]
git-tree-sha1 = "794c760e6aafe9f40dcd7dd30526ea33f0adc8b7"
uuid = "c0aeaf25-5076-4817-a8d5-81caf7dfa961"
version = "1.15.1"
weakdeps = ["SparseArrays", "StaticArraysCore"]

    [deps.SciMLOperators.extensions]
    SciMLOperatorsSparseArraysExt = "SparseArrays"
    SciMLOperatorsStaticArraysCoreExt = "StaticArraysCore"

[[deps.SciMLPublic]]
git-tree-sha1 = "0ba076dbdce87ba230fff48ca9bca62e1f345c9b"
uuid = "431bcebd-1456-4ced-9d72-93c2757fff0b"
version = "1.0.1"

[[deps.SciMLStructures]]
deps = ["ArrayInterface", "PrecompileTools"]
git-tree-sha1 = "607f6867d0b0553e98fc7f725c9f9f13b4d01a32"
uuid = "53ae85a6-f571-4167-b2af-e1d143709226"
version = "1.10.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "2700b235561b0335d5bef7097a111dc513b8655e"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.7.2"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "4f96c596b8c8258cc7d3b19797854d368f243ddc"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.4"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "aceda6f4e598d331548e04cc6b2124a6148138e3"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.10"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "91f091a8716a6bb38417a6e6f274602a19aaa685"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.5.2"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "9297459be9e338e546f5c4bedb59b3b5674da7f1"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.6.2"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.SymbolicIndexingInterface]]
deps = ["Accessors", "ArrayInterface", "RuntimeGeneratedFunctions", "StaticArraysCore"]
git-tree-sha1 = "94c58884e013efff548002e8dc2fdd1cb74dfce5"
uuid = "2efcf032-c050-4f8e-a9bb-153293bab1f5"
version = "0.3.46"

    [deps.SymbolicIndexingInterface.extensions]
    SymbolicIndexingInterfacePrettyTablesExt = "PrettyTables"

    [deps.SymbolicIndexingInterface.weakdeps]
    PrettyTables = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "3748bd928e68c7c346b52125cf41fff0de6937d0"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.29"

    [deps.TimerOutputs.extensions]
    FlameGraphsExt = "FlameGraphs"

    [deps.TimerOutputs.weakdeps]
    FlameGraphs = "08572546-2f56-4bcf-ba4e-bab62c3a3f89"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.TruncatedStacktraces]]
deps = ["InteractiveUtils", "MacroTools", "Preferences"]
git-tree-sha1 = "ea3e54c2bdde39062abf5a9758a23735558705e1"
uuid = "781d530d-4396-4725-bb49-402e4bee1e77"
version = "1.4.0"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "96478df35bbc2f3e1e791bc7a3d0eeee559e60e9"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.24.0+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "9cce64c0fdd1960b597ba7ecda2950b5ed957438"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.2+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "b5899b25d17bf1889d25906fb9deed5da0c15b3b"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.12+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a4c0ee07ad36bf8bbce1c3bb52d21fb1e0b987fb"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.7+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "a376af5c7ae60d29825164db40787f15c80c7c54"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.3+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "a5bc75478d323358a90dc36766f3c99ba7feb024"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.6+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "aff463c82a773cb86061bce8d53a0d976854923e"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.5+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "4909eb8f1cbf6bd4b1c30dd18b2ead9019ef2fad"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.18.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "e3150c7400c41e207012b41659591f083f3ef795"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.3+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "9750dc53819eba4e9a20be42349a6d3b86c7cdf8"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.6+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f4fc02e384b74418679983a97385644b67e1263b"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll"]
git-tree-sha1 = "68da27247e7d8d8dafd1fcf0c3654ad6506f5f97"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "44ec54b0e2acd408b0fb361e1e9244c60c9c3dd4"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "5b0263b6d080716a02544c55fdff2c8d7f9a16a0"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.10+0"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f233c83cad1fa0e70b7771e0e21b061a116f2763"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.2+0"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "801a858fc9fb90c11ffddee1801bb06a738bda9b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.7+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "00af7ebdc563c9217ecc67776d1bbf037dbcebf4"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.44.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c3b0e6196d50eab0c5ed34021aaa0bb463489510"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.14+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "371cc681c00a3ccc3fbc5c0fb91f58ba9bec1ecf"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.13.1+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "125eedcb0a4a0bba65b657251ce1d27c8714e9d6"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.4+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libdrm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "63aac0bcb0b582e11bad965cef4a689905456c03"
uuid = "8e53e030-5e6c-5a89-a30b-be5b7263a166"
version = "2.4.125+1"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56d643b57b188d30cccc25e331d416d3d358e557"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.13.4+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "91d05d7f4a9f67205bd6cf395e488009fe85b499"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.28.1+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "6ab498eaf50e0495f89e7a5b582816e2efb95f64"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.54+0"

[[deps.libva_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "libdrm_jll"]
git-tree-sha1 = "7dbf96baae3310fe2fa0df0ccbb3c6288d5816c9"
uuid = "9a156e7d-b971-5f62-b2c9-67348b8fb97c"
version = "2.23.0+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b4d631fd51f2e9cdd93724ae25b2efc198b059b1"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "1350188a69a6e46f799d3945beef36435ed7262f"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2022.0.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "a1fc6507a40bf504527d0d4067d718f8e179b2b8"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.13.0+0"
"""

# ╔═╡ Cell order:
# ╠═d44629e2-07a2-11f1-862d-ffeb4714407a
# ╟─f6f23886-7df7-41ea-878c-131198ba9f0e
# ╠═5a030e89-08a3-4ec3-958e-74a084a96a32
# ╠═63f4cb2d-768b-4154-8cfc-5a174046ce1c
# ╠═a33fbec9-4b51-47ca-872c-3c70e0986b20
# ╠═9b364800-93e9-48fe-a0cd-efcbd9c3f398
# ╠═cb529e73-d589-4b5e-8799-a01a9fa60f23
# ╠═9760e4cd-5790-4843-b7b5-ba2d0190fa37
# ╟─5ccef611-c8bf-4124-820d-206e571f24fb
# ╟─9703f6fb-dab5-46aa-af14-6600184b2a1f
# ╟─53229c8f-3f5c-4195-9717-195a536c1dc6
# ╟─cab6acf0-ddbf-4df9-8a7b-9af7a46159bc
# ╟─daa960d0-b282-467f-8a93-43ed71a5b84f
# ╟─1eed3115-ba8f-4a68-ab2b-9a8e66139d81
# ╟─d8803f0c-670e-4a00-8ed3-45ae8441e797
# ╠═11dd1844-e15a-4043-a342-449c098abf2b
# ╠═94b0f311-5784-4e2e-b2c6-36cc3f163207
# ╠═805fe0ac-2acb-44d6-8ed5-b5d3543a8f03
# ╠═122adeac-702e-4dc8-8098-d216168984ca
# ╠═4d60b037-1a21-49bf-aae7-8e79d08ae623
# ╠═49b29633-eca2-46ee-aa39-842a4504759c
# ╟─b3f79f29-3488-45d2-b839-4dcfe20e1eaf
# ╠═cb26d768-2a0f-4581-96e1-205b98bd4981
# ╠═4bde3946-fdf8-4bf2-8888-92eef8eda718
# ╠═135ee28d-74cf-41b0-b41c-8d15f712af16
# ╟─45ceae60-6686-47f4-9f38-fc0d349921a2
# ╟─ac439b9a-7b25-45e5-8c70-624ab88b6359
# ╟─cbbf7c1f-f0b7-4422-8516-1ded7c6a596f
# ╟─6bba2ae7-889d-41a1-a414-e5800361a573
# ╟─14778872-d513-44c2-9379-0e7d1c7a5fd6
# ╠═80b53ae2-0e59-4b4f-9c4b-13de29eaed5d
# ╠═1d55f0e2-adc1-4a7c-aa6c-9fbd721d117b
# ╠═5fd4d32b-995e-458a-9a7f-c12de9a6327f
# ╠═bc8fa8f7-8c66-4108-9a74-f1587d944c99
# ╠═05496e2c-04ae-4467-bb50-7f692968f3fb
# ╟─505a6e03-e123-4253-b719-971022018d3b
# ╟─f9e281cf-0c30-40a6-9d06-e257f2d78df5
# ╠═6c06982d-ac87-45b0-a885-92efdf325aea
# ╠═51cb4863-772e-4784-830d-cccbb37e891f
# ╟─5f798ff3-f463-4bbe-9c31-b08ffb9d6d37
# ╠═1d21d0fd-6e47-4913-b05f-662b2d04d727
# ╟─5b4ec8f0-07d6-45b9-a20c-30b52271f359
# ╟─723a4485-c64e-48fa-907e-3c5ea0316023
# ╟─3287fb59-f3ce-4763-8715-7457418ef7be
# ╟─739fc6c8-3305-4715-bc45-7741489cb63e
# ╟─144cff2f-bf84-460d-b17c-433fc3bfd614
# ╠═2951df38-84e4-45a9-b75c-d8f56b87d23b
# ╠═8317da7d-d6b1-4d4a-82cf-389f85eef719
# ╟─a4078281-1160-4455-a9fc-97df9c2b8062
# ╠═debe9624-fb48-4633-a94f-b457040eb6c0
# ╠═5774aba4-8294-46fb-9623-c0c165f08301
# ╠═ca182723-485a-4664-b2ee-d054b10379a6
# ╠═10f46658-3ac6-49c6-9820-d7a56ef6db6c
# ╠═6f729716-7ac9-427d-9df0-9f869947f322
# ╠═64c2ed54-2ef5-4b4f-8464-497f1a8ffce1
# ╠═1749ea8b-1c36-48b2-841d-d00f8e482d6f
# ╟─b12a629d-fe8c-426a-8f0a-83fa93055935
# ╟─3b19c567-c772-47b8-a671-62420f48800e
# ╟─b8fd5f9e-d2b5-40f0-9025-d0db531164c3
# ╠═a2ba4ab8-d599-4785-87bd-b38c9337f370
# ╠═9e7fb844-d442-4ec9-bbbe-84960c5c4130
# ╠═5f4869d4-f3dd-49fc-8e5f-56e1ab0be2e7
# ╠═639d9fea-848b-4212-9df7-8a98fa2e2a04
# ╟─3e1f5517-1cb1-4d25-afc6-ade63999e48f
# ╟─928a29e7-af2a-419e-b2b4-7650fa3fb41e
# ╟─37ed9b7a-3294-4b05-969e-052e83997e7c
# ╠═97fb16d6-c3f2-461b-8ffe-478674913d86
# ╠═818719aa-b636-4db4-859c-3bbd30e23317
# ╠═f92b206c-786d-4e75-b402-80217f33ddca
# ╠═fb04e075-a96b-4365-b7aa-ee0a8db7285c
# ╠═0f8e6aa3-acf6-4882-a0d7-e742a221acea
# ╠═c16d355c-0e47-4c1d-9f48-894e3817422c
# ╠═d036ef8e-36d4-4445-9af5-4c8a719cbe64
# ╠═bb72b814-6f07-4109-8f61-9cfe833d834e
# ╠═dd2b7266-db26-4605-b1da-69b91adf7856
# ╟─bf57d43a-aede-43ac-9572-55630bbae190
# ╠═aa4b97e0-377e-42b5-9d5b-27e2b0027143
# ╟─1521b9aa-40e8-4687-a865-e733a9c71d6e
# ╟─47ae380b-3329-4b24-a9e2-288ec1c512aa
# ╟─71c01fb0-80b3-4e2f-a527-316fd992e2d5
# ╟─a276362c-4c11-42bb-bfd7-02f52bd68bb0
# ╟─e8a7cf79-3e5d-4f0a-98cc-af7eba1bdbd2
# ╠═16f7c3f2-d836-41da-bac8-ac0ba1b6fdfb
# ╟─bc0e23f9-a5a1-4848-bbc2-ab467d1e4f09
# ╠═5ed243c0-104b-4f68-bcba-348fe5deba97
# ╠═1f6073ab-5c5f-4caa-b8a2-f568f2c2dbcd
# ╠═f6b6010e-032f-49ed-bde8-11f1df819885
# ╠═d7642bbd-1937-47ed-9686-591ded4ad0f8
# ╟─87a8be99-30eb-4246-8bc7-0e0a5f6b6fd9
# ╠═f048c295-56a9-47c6-9198-2ce09b3d6982
# ╟─f21752a2-afdb-4c4c-b8b6-faeb396a23de
# ╠═d021c01d-9a91-4900-b2e7-f94442870766
# ╟─d8067735-c715-4f65-932f-7d6d8ed49a51
# ╟─83910760-e461-43be-87d7-7e8ebb154731
# ╠═af2d6c23-211c-4669-bacd-5f2267b01e38
# ╠═8eb2bea0-dfc9-4be1-80b0-cd14fa587213
# ╠═f8d84352-9bc2-4fe9-81c9-411fa0c3d42c
# ╠═46270c42-11a9-4c0b-a397-1baf4ba73c71
# ╟─4afbadf8-e387-46a0-9903-e077f9f7b5f6
# ╠═2e86c3ab-b3e7-4faa-afa7-1edd7a26296c
# ╠═d03f4fc4-ccff-4d61-8146-5fe890037cee
# ╠═e943a7bc-ac2f-48e3-9970-bb84880ede93
# ╠═ac7cf451-9b31-4f4b-9c1a-1c398e42a0a8
# ╠═c26ae70a-55b3-4928-b270-ef128c8c77eb
# ╠═b3806c7c-e849-4386-92d8-295fdea4aa44
# ╠═325a6250-5b68-46af-9ad7-08a4645af429
# ╠═d19f75d6-daa3-4ace-961e-aa42c239a464
# ╠═3b1819df-0fab-4e8b-8974-72409e287588
# ╠═ffb93cdf-d4dd-4c66-b88d-bad81d748681
# ╠═991a397f-e72b-4271-a182-4b0085aa667a
# ╟─ff5ed656-0d26-42a5-bff4-8d6b709eb8e6
# ╠═04ac60fd-ec95-49cd-aea4-2125f8722dc6
# ╠═a186a66d-d99a-49db-bfd7-24c18513eeec
# ╟─8ac244cb-3f7c-45bc-9021-d535dc00fed8
# ╟─0c2af742-fd02-4e44-bed4-e1373facd206
# ╠═c5293299-c0a4-4233-aeca-6d6267bd6516
# ╠═4ac02b34-4f0f-47c5-be2a-3646c7347290
# ╟─dc130e84-4719-4aa7-894a-bb020a1224ae
# ╟─1110f7f3-c508-4db1-b2fb-59db33548b01
# ╟─8c1f2c5a-a59f-431f-b1f8-fba0519c97f2
# ╟─8dea93f9-f3fc-4e6a-998c-0372c94fe640
# ╠═44c27737-0c9e-4cfd-866c-73d70ecb8356
# ╠═bd810cdd-aa6f-4fe1-9427-ed574f9bf31d
# ╠═795e4e12-9f8a-4a3a-8554-cb297bdb60f1
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
