using BenchmarkTools
using Printf

global compfun = nothing
global compfunargs = nothing

# Store benchmark results here
global benchtime = Dict{Union{Function, String}, Float64}()

"Compare benchmark times"
function compare(fun, ref::Union{Function, String}, funargs;
                    verbose=true, reverse=true, verbose2=true,
                    samples=1000, seconds=1)

    global compfun = fun
    global compfunargs = funargs

    BenchmarkTools.DEFAULT_PARAMETERS.samples = samples
    BenchmarkTools.DEFAULT_PARAMETERS.seconds = seconds

    bmark = @benchmark compfun(compfunargs...)
    benchtime[fun] = minimum(bmark.times)/1e6

    if verbose
        printcomparison(fun, ref, reverse)
    end
    if verbose2
        io = IOBuffer()
        show(io, "text/plain", bmark)
        println("\n"*String(take!(io)))
    end
end

function printcomparison(fun, ref, reverse)
    ratio = benchtime[fun]/benchtime[ref]
    if reverse
        @printf "%-20s is %5.2f times faster than %20s (%5.3fms vs %5.3fms)\n" fun 1/ratio ref benchtime[fun] benchtime[ref]
    else
        @printf "%-20s is %5.2f times faster than %20s (%5.3fms vs %5.3fms)\n" ref ratio fun benchtime[ref] benchtime[fun]
    end
end




"Generates an array of particles of type `PType` arranged in a nxnxn box"
function generate_particles(PType, n, lambda; l=1, Gamma=ones(3))

    sigma = l/(n-1)*lambda
    particles = fill(zero(PType), n^3)

    xs = range(0, stop=l, length=n)

    # Adds particles in a regular lattice
    ind = 1
    for k in 1:n
        for j in 1:n
            for i in 1:n
                X = [xs[i], xs[j], xs[k]]
                particles[ind] = PType(X, Gamma, sigma)
                ind += 1
            end
        end
    end

    return particles
end
