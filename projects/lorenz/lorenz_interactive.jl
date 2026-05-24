using GLMakie, DataStructures

Base.@kwdef mutable struct Lorenz
    dt::Float64 = 0.01
    σ::Float64 = 10
    ρ::Float64 = 28
    β::Float64 = 8/3
    x::Float64 = 1
    y::Float64 = 1
    z::Float64 = 1
end

function step!(l::Lorenz)
    dx = l.σ * (l.y - l.x)
    dy = l.x * (l.ρ - l.z) - l.y
    dz = l.x * l.y - l.β * l.z 
    l.x += l.dt * dx 
    l.y += l.dt * dy
    l.z += l.dt * dz
    Point3f(l.x, l.y, l.z)
end

attractor = Lorenz()
points = Observable(CircularBuffer{Point3f}(500))
colors = Observable(Int[])

set_theme!(theme_black())

fig, ax, l = lines(points, color = colors,
    colormap = :inferno, transparency=true,
    axis = (; type=Axis3, protrusions=(0,  0, 0, 0),
            viewmode=:fit, limits=(-30, 30, -30, 30, 0, 50)))

menu = Menu(fig, options = ["viridis", "inferno", "magma", "plasma"], default="inferno")

on(menu.selection) do cmap
    l.colormap = cmap
end
notify(menu.selection)

display(fig)

# hidedecorations!(ax)
# hidespines!(ax)

run = Button(fig[2, 1]; label="run", tellwidth=false)
isrunning = Observable(false)
on(run.clicks) do clicks; isrunning[] = !isrunning[]; end
on(run.clicks) do clicks
    @async while isrunning[]
        isopen(fig.scene) || break
          push!(points[], step!(attractor))
          push!(colors[], min(length(points[]), 100))
          l.colorrange = (0, min(length(points[]), 100))
          colors[] = colors[]
          points[] = points[]
          sleep(0.001)
    end
end

# while true
#     push!(points[], step!(attractor))
#     push!(colors[], min(length(points[]), 100))
#     l.colorrange = (0, min(length(points[]), 100))
#     colors[] = colors[]
#     points[] = points[]
#     sleep(0.001)
# end
