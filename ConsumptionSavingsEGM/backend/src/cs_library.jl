using LinearAlgebra



function get_log_agrid(Na::Int, ϕ::Float64, amax::Float64;
                             ϵ_shift::Float64 = 1e-6)
    # a ∈ [-ϕ, amax]
    amin = -ϕ

    # shift so grid is positive before taking logs
    amin_tilde = amin + ϕ + ϵ_shift         # ≈ ϵ_shift
    amax_tilde = amax + ϕ + ϵ_shift

    # equally spaced in log(ã)
    log_min = log(amin_tilde)
    log_max = log(amax_tilde)
    log_grid = range(log_min, log_max, length = Na)
    a_tilde = exp.(log_grid)

    # shift back
    agrid = a_tilde .- ϕ .- ϵ_shift
    return agrid
end





"""
    gauss_hermite(n::Int)

Return nodes `x` and weights `w` for n-point Gauss–Hermite quadrature
with weight function exp(-x^2) on (-∞, ∞).

The rule satisfies
    ∫_{-∞}^{∞} f(x) * exp(-x^2) dx ≈ ∑ w[i] * f(x[i])
"""
function gauss_hermite(n::Int)
    # symmetric tridiagonal matrix with sqrt(1:(n-1)) on off-diagonals
    d  = zeros(n)                 # diagonal
    sd = sqrt.(1:(n-1))           # sub/super-diagonal

    T = SymTridiagonal(d, sd)
    ev = eigen(T)

    x = ev.values                 # nodes
    v1 = ev.vectors[1, :]         # first row of eigenvectors
    w = sqrt(pi) .* (v1 .^ 2)     # weights for exp(-x^2) rule

    return x, w
end

"""
    make_quadrature(Nε::Int)

Construct quadrature nodes `εnodes` and probability weights `wε`
for expectations with respect to a standard normal N(0,1).

The expectation
    E[g(Z)]  (Z ~ N(0,1))
is approximated as
    ∑ wε[i] * g(εnodes[i])
"""
function make_quadrature(Nε::Int)
    x, w = gauss_hermite(Nε)

    # Convert Hermite weights to probability weights for N(0,1)
    # For Z ~ N(0,1), we have:
    #   E[g(Z)] = (1/√π) ∫ g(x/√2) * exp(-x^2) dx
    # so we use nodes ε = x / √2 and weights wε = w / √π.
    εnodes = x ./ sqrt(2.0)
    wε = w ./ sqrt(pi)

    # normalize to guard against tiny numerical drift
    wε ./= sum(wε)

    return εnodes, wε
end

#=
function init_policy!(cs::ConsSavEGMCUDA)
    Na, Ny = cs.Na, cs.Ny
    for ja in 1:Na, jy in 1:Ny
        cs.ga[ja, jy] = cs.agrid[ja]    # a' = a as a simple guess
    end
    return
end
=#
struct ConsSavCPUEGM
    β  :: Float64
    γ  :: Float64
    R  :: Float64
    ϕ  :: Float64
    ρ  :: Float64
    σ  :: Float64

    Na :: Int
    Ny :: Int
    Nε :: Int

    apgrid :: Vector{Float64}
    ygrid  :: Vector{Float64}

    a_endo :: Matrix{Float64}
    c_endo :: Matrix{Float64}
    muc    :: Matrix{Float64}
    ga     :: Matrix{Float64}
    gc     :: Matrix{Float64}
    V      :: Matrix{Float64}
end

function to_cpu(cs::ConsSavEGMCUDA)
    ConsSavCPUEGM(
        cs.β, cs.γ, cs.R, cs.ϕ, cs.ρ, cs.σ,
        cs.Na, cs.Ny, cs.Nε,
        Array(cs.apgrid),
        Array(cs.ygrid),
        Array(cs.a_endo),
        Array(cs.c_endo),
        Array(cs.muc),
        Array(cs.ga),
        Array(cs.gc),
        Array(cs.V),
    )
end


"""
    get_jyp(ypv, ygrid, Ny)

Return the index jyp in 1:Ny such that ygrid[jyp] is
the closest point in ygrid to ypv (nearest neighbor).
Intended for use inside GPU kernels.
"""
@inline function get_jyp(ypv::Float64, ygrid, Ny::Int)
    jyp_best = 1
    dist_min = abs(ygrid[1] - ypv)

    @inbounds for jy in 2:Ny
        d = abs(ygrid[jy] - ypv)
        if d < dist_min
            dist_min = d
            jyp_best = jy
        end
    end

    return jyp_best
end

@inline function get_jap(a_endo, av::Float64, jy::Int, Na::Int)
    jap_best = 1
    dist_min = abs(a_endo[1, jy] - av)

    @inbounds for jap in 2:Na
        d = abs(a_endo[jap, jy] - av)
        if d < dist_min
            dist_min = d
            jap_best = jap
        end
    end

    return jap_best
end

@inline function interp_y_from_nearest(gc, jap::Int, ypv::Float64, ygrid, Ny::Int)
    jyp = get_jyp(ypv, ygrid, Ny)

    jL = jyp
    jH = jyp

    if ypv < ygrid[jyp] && jyp > 1
        jL = jyp - 1
        jH = jyp
    elseif ypv > ygrid[jyp] && jyp < Ny
        jL = jyp
        jH = jyp + 1
    end

    if jL == jH
        return gc[jap, jL]
    end

    yL = ygrid[jL]
    yH = ygrid[jH]

    if yH == yL
        return gc[jap, jL]
    end

    wH = (ypv - yL) / (yH - yL)
    wL = 1.0 - wH

    c_star = wL * gc[jap, jL] + wH * gc[jap, jH]
    return c_star
end