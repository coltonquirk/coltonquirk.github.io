using GLMakie, DataStructures, LaTeXStrings


# Body structures 
# m = [m1, m2, ..., mn]
# bodies = [[x1, y1, z1, vx1, vy1, vz1],
#           [x2, y2, z2, vx2, vy2, vz2]]

m = [1, 1]
bodies = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
          [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0]]


function force(m, bodies)
    n = length(m)
    acc = zeros(n, 3)

    # Figure out how to use this iteration trick
    for i in 1:n
        for j in 1:i
            r1 = bodies[i][1:3]
            r2 = bodies[j][1:3]
            
            force = - G * m[j] * (r2 - r1) / norm(r2 - r1)^3

            acci = force / m[i]
            accj = -force / m[j] 

            acc[i] += acci 
            acc[j] += accj
        end
    end

    return acc
end

force(m, bodies)
