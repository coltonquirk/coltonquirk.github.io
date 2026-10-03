using GLMakie, DataStructures, ColorSchemes, Colors, LaTeXStrings

Base.@kwdef mutable struct Lorenz
    # parameters
    dt::Float64 = 1e-2
    σ::Float64  = 10
    ρ::Float64  = 28
    β::Float64  = 8/3
    # Initial conditions
    x::Float64  = 1
    y::Float64  = 1
    z::Float64  = 1
end

function step!(l::Lorenz)
    # Differential equations
    dx = l.σ * (l.y - l.x)
    dy = l.x * (l.ρ - l.z) - l.y
    dz = l.x * l.y - l.β * l.z 
    # Euler step
    l.x += l.dt * dx 
    l.y += l.dt * dy 
    l.z += l.dt * dz
    # Return an actual 3d point
    Point3f(l.x, l.y, l.z)
end

eps = 1e-8 
num = 10 
attractors = [
    Lorenz(z=1+(n-1)*eps)
    for n in 1:num
]

npoints = 300
all_points = [
    Observable(CircularBuffer{Point3f}(npoints)) 
    for n in 1:num
]
all_colors = [
    Observable(Float32[])
    for n in 1:num
]

function fade_colormap(color::Colorant, n=256)
    r, g, b = red(color), green(color), blue(color)
    colors = [RGBA(r, g, b, α^5) for α in LinRange(0f0, 1f0, n)]
    return ColorScheme(colors)
end
blues = [get(ColorSchemes.Blues, t) for t in LinRange(0.5, 0.8, num)]
cmaps = [fade_colormap(c) for c in blues]

dark_latex_theme = merge(theme_dark(), theme_latexfonts())
set_theme!(dark_latex_theme)

fig = Figure()
ax = Axis3(fig[1, 2],
            protrusions=(0, 0, 0, 0),
            viewmode=:fit, limits=(-30, 30, -30, 30, 0, 50)
)
Label(fig[1, :, Top()], "Lorenz Attractor",
    fontsize=30,
    color=:white,
    valign=:bottom,
)
for (i, points) in enumerate(all_points)
    lines!(ax, points; color=all_colors[i], colormap=cmaps[i], transparency=true)
end

eq_layout = fig[1, 1] = GridLayout(tellwidth=false, tellheight=false)

Label(eq_layout[1, 1], L"\frac{\mathrm{d}x}{\mathrm{d}t} = \sigma(y - x)",   fontsize=30, color=:white, halign=:left) 
Label(eq_layout[2, 1], L"\frac{\mathrm{d}y}{\mathrm{d}t} = x(\rho - z) - y", fontsize=30, color=:white, halign=:left)
Label(eq_layout[3, 1], L"\frac{\mathrm{d}z}{\mathrm{d}t} = xy - \beta z",    fontsize=30, color=:white, halign=:left)

function reset!()
    for i in 1:num 
        empty!(all_points[i][])
        all_points[i][] = all_points[i][]
        all_colors[i][] = Float32[]
        attractors[i] = Lorenz(z=1+(i-1)*eps)
    end
end


button_layout = fig[2, 1:2] = GridLayout()
run = Button(button_layout[2, 1]; label="run")
reset = Button(button_layout[2, 2]; label="reset")

rowsize!(fig.layout, 2, Fixed(40))

on(reset.clicks) do _ 
    isrunning[] = false 
    sleep(0.05) 
    reset!()
end


isrunning = Observable(false)
on(run.clicks) do clicks; isrunning[] = !isrunning[]; end
on(run.clicks) do clicks
    @async while isrunning[]
        isopen(fig.scene) || break 
        for (attractor, points, colors) in zip(attractors, all_points, all_colors)
            push!(points[], step!(attractor))
            n = length(points[])
            colors[] = n <= 1 ? Float32[1f0] : collect(LinRange(0f0, 1f0, n))
            points[] = points[]
        end
        sleep(8e-3)
    end
end

fig
