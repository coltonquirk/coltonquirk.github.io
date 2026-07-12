using GLMakie, LaTeXStrings, LinearAlgebra, DataStructures

const G = 1.0 # natural units

# @Base.kwdef mutable struct Planet 
#     m::Float64 = 1.0
#     pos::Vector{Float64} = [1.0, 0.0, 0.0]
#     vel::Vector{Float64} = [0.0, 1.0, 0.0]
# end

# function force(t, p1, p2)
#     fReturn =  - G * p1.m * p2.m (p2.pos - p1.pos) / norm(p2.pos - p1.pos)^3
#     return fReturn
# end

function force(t, first, second)
    r1 = first[1:3]; v1 = first[4:6]
    r2 = second[1:3]; v2 = second[4:6]

    acc = - G * (r2 - r1) / norm(r2 - r1)^3

    fReturn = [[v1, acc],
                [v2, -acc]]
    return fReturn
end


function rk4(first, second, f, dt)
    acc11, acc12 = f(first, second)
    acc21, acc22 = f(first+acc11*0.5, second-acc12*0.5)
    acc31, acc32 = f(first+acc21*0.5, second-acc22*0.5)
    acc41, acc42 = f(first+acc31*0.5, second-acc32*0.5)

    return dt * (acc11 + acc21*2.0 + acc31 * 2.0 + acc41) / 6.0
end

# planet structure
# could extend to include masses?
# [[x1, y1, z1, vx1, vy1, vz1],
#  [x2, y2, z2, vx2, vx2, vy2]]

function nbody!(planets)
    dt = 1e-4
    forces = zero(planets)    

    for (i, planet1) in enumerate(eachrow(planets))
        for (j, planet2) in enumerate(eachrow(planets))
            if i != j && i > j 
                force_between = rk4(planet1[1], planet2[1], force, dt) 
                forces[i] += force_between
                forces[j] -= force_between
            end
        end
    end

    planets += forces
end

function display_nbody()
    n_planets = 2
    planets = [[1, 0, 0, 0, 1, 0],
               [-1, 0, 0, -1, 0]]

    pos1 = Observable(Point2f(1, 0))
    pos2 = Observable(Point2f(-1, 0))

    fig = Figure()
    ax = Axis(fig[1,1], title="n-body sim")

    set_theme!(theme_dark())

    scatter!(ax, pos1)
    scatter!(ax, pos2)
    for _ in 1:50
        nbody!(planets)
        pos1[] = Point2f(planets[1][1], planets[1][2])
        pos2[] = Point2f(planets[1][1], planets[1][2])
        sleep(1/60)
    end
end

display_nbody()
