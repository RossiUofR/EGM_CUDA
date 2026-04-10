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

    kgrid  :: Vector{Float64}
    zgrid  :: Vector{Float64}
    Pz     :: Matrix{Float64}

    # Policies on fixed grid
    gk     :: Matrix{Float64}
    gc     :: Matrix{Float64}

    # EGM work arrays
    k_endo :: Matrix{Float64}
    c_endo :: Matrix{Float64}
    muc    :: Matrix{Float64}
end


function to_cpu(gw::GrowthCUDA)
    GrowthCPU(
        gw.β, gw.σ, gw.α, gw.δ,
        gw.Nk, gw.Nz,
        Array(gw.kgrid),
        Array(gw.zgrid),
        Array(gw.Pz),
        Array(gw.gk),
        Array(gw.gc),
        Array(gw.k_endo),
        Array(gw.c_endo),
        Array(gw.muc),
    )
end
# For given (k', z, c_t), find jk that makes budget-implied c closest to c_t
@inline function get_k_from_cv(kgrid, zv::Float64, kvp::Float64,
                               c_t::Float64, α::Float64, δ::Float64,
                               Nk::Int)
    kL = kgrid[1]
    fL = (1.0 - δ) * kL + zv * kL^α - kvp - c_t

    if fL >= 0.0
        return kL
    end

    @inbounds for jk in 2:Nk
        kH = kgrid[jk]
        fH = (1.0 - δ) * kH + zv * kH^α - kvp - c_t

        if fL <= 0.0 && fH >= 0.0
            if fH == fL
                return kL
            end

            wH = -fL / (fH - fL)
            wL = 1.0 - wH
            return wL * kL + wH * kH
        end

        kL = kH
        fL = fH
    end

    return kgrid[Nk]
end

