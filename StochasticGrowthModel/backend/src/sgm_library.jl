using CUDA, Distributions

function get_jkp(k_endo, kv::Float64, jz::Int, Nk::Int)
    best_j = 1
    best_d = abs(k_endo[1, jz] - kv)
    @inbounds for j in 2:Nk
        d = abs(k_endo[j, jz] - kv)
        if d < best_d
            best_d = d
            best_j = j
        end
    end
    return best_j
end

function fill_guess!(gk, gc, kgrid, zgrid,
                     α::Float64, δ::Float64,
                     Nk::Int, Nz::Int)

    jk = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    jz = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if jk > Nk || jz > Nz
        return
    end

    kv = kgrid[jk]
    zv = zgrid[jz]

    # simple rule: k′ = (1-δ)k, then compute c from budget
    kvp = (1.0 - δ) * kv
    yv  = output(kv, zv, α)
    cv  = (1.0 - δ) * kv + yv - kvp

    gk[jk, jz] = kvp
    gc[jk, jz] = max(cv, 1e-10)

    return
end

function init_policy!(gw::GrowthCUDA)
    Nk, Nz = gw.Nk, gw.Nz
    threads = (16, 16)
    blocks  = (cld(Nk, threads[1]), cld(Nz, threads[2]))

    @cuda threads=threads blocks=blocks fill_guess!(
        gw.gk, gw.gc,
        gw.kgrid, gw.zgrid,
        gw.α, gw.δ,
        Nk, Nz
    )
end

function tauchenlib(N::Int, ρ::Float64, σϵ::Float64; m::Float64 = 3.0)
    σz   = σϵ / sqrt(1 - ρ^2)
    zmax = m * σz
    zmin = -zmax
    zgrid = collect(range(zmin, zmax, length = N))
    step  = zgrid[2] - zgrid[1]

    P = zeros(N, N)
    dist = Normal()

    for j in 1:N
        for k in 1:N
            if k == 1
                P[j, k] = cdf(dist, (zgrid[1] - ρ*zgrid[j] + step/2) / σϵ)
            elseif k == N
                P[j, k] = 1 - cdf(dist, (zgrid[N] - ρ*zgrid[j] - step/2) / σϵ)
            else
                ub = (zgrid[k] - ρ*zgrid[j] + step/2) / σϵ
                lb = (zgrid[k] - ρ*zgrid[j] - step/2) / σϵ
                P[j, k] = cdf(dist, ub) - cdf(dist, lb)
            end
        end
    end

    # Simple sanity checks
    row_sums = sum(P, dims = 2)
    max_dev  = maximum(abs.(row_sums .- 1.0))
    if max_dev > 1e-10
        @warn "Tauchen: row sums deviate from 1 by up to $(max_dev)"
    end

    return zgrid, P
end

struct GrowthCPU
    β  :: Float64
    σ  :: Float64
    α  :: Float64
    δ  :: Float64

    Nk :: Int
    Nz :: Int

    kgrid :: Vector{Float64}
    zgrid :: Vector{Float64}

    gk :: Matrix{Float64}
    gc :: Matrix{Float64}
end

function to_cpu(gw::GrowthCUDA)
    GrowthCPU(
        gw.β, gw.σ, gw.α, gw.δ,
        gw.Nk, gw.Nz,
        Array(gw.kgrid),
        Array(gw.zgrid),
        Array(gw.gk),
        Array(gw.gc),
    )
end

# For given (k', z, c_t), find jk that makes budget-implied c closest to c_t
function get_jk_from_cv(kgrid, zv::Float64, kvp::Float64,
                        c_t::Float64, α::Float64, δ::Float64,
                        Nk::Int)
    best_jk = 1
    kv     = kgrid[1]
    yv     = output(kv, zv, α)
    c_best = (1.0 - δ) * kv + yv - kvp
    best_d = abs(c_best - c_t)

    @inbounds for jk in 2:Nk
        kv = kgrid[jk]
        yv = output(kv, zv, α)
        c_candidate = (1.0 - δ) * kv + yv - kvp
        d = abs(c_candidate - c_t)
        if d < best_d
            best_d = d
            best_jk = jk
        end
    end

    return best_jk
end


# Given k and z index jz, find nearest endogenous index j0
# and a small symmetric neighborhood around it
function get_neighborhood_indices(k_endo, kv::Float64, jz::Int,
                                  Nk::Int; halfwidth::Int = 1)
    # nearest endogenous index (you already have this)
    j0 = get_jkp(k_endo, kv, jz, Nk)

    jmin = max(j0 - halfwidth, 1)
    jmax = min(j0 + halfwidth, Nk)

    idxs = Int[]
    @inbounds for j in jmin:jmax
        push!(idxs, j)
    end

    return j0, idxs
end


# 1) Build distance-based scores in k-space and center them
function k_distance_scores!(scores, k_endo, idxs, kv::Float64,
                            jz::Int, σk::Float64)
    n = length(idxs)
    @inbounds for p in 1:n
        j = idxs[p]
        k_end = k_endo[j, jz]
        d = (k_end - kv) / σk
        scores[p] = -d^2             # higher score for closer k_end
    end

    # subtract max for numerical stability
    s_max = scores[1]
    @inbounds for p in 2:n
        if scores[p] > s_max
            s_max = scores[p]
        end
    end
    @inbounds for p in 1:n
        scores[p] -= s_max
    end

    return
end


# 2) Convert centered scores to softmax weights with temperature τ
function softmax_from_scores!(weights, scores, n::Int, τ::Float64)
    sum_exp = 0.0
    @inbounds for p in 1:n
        w = exp(scores[p] / τ)
        weights[p] = w
        sum_exp += w
    end

    inv_sum = 1.0 / sum_exp
    @inbounds for p in 1:n
        weights[p] *= inv_sum
    end

    return
end


# Convenience wrapper: from k_endo and (k,z) to softmax weights over idxs
function local_softmax_from_k!(weights, k_endo, idxs, kv::Float64,
                               jz::Int, σk::Float64, τ::Float64)
    n = length(idxs)
    scores = zeros(Float64, n)
    k_distance_scores!(scores, k_endo, idxs, kv, jz, σk)
    softmax_from_scores!(weights, scores, n, τ)
end


# Smooth c at (k,z) using neighbors and their softmax weights
function smooth_c_from_neighbors(c_endo, idxs, jz::Int, weights)
    n = length(idxs)
    c_sm = 0.0
    @inbounds for p in 1:n
        j = idxs[p]
        c_sm += weights[p] * c_endo[j, jz]
    end
    return c_sm
end

function compute_sigma_k(gw::GrowthCUDA; multiplier::Float64 = 2.0)
    k_endo_cpu = Array(gw.k_endo)
    Δs = Float64[]
    for jz in 1:gw.Nz
        col = k_endo_cpu[:, jz]
        @inbounds for j in 1:(gw.Nk - 1)
            push!(Δs, col[j+1] - col[j])
        end
    end
    Δbar = median(Δs)
    return multiplier * Δbar
end