struct GrowthCUDA
    β  :: Float64
    σ  :: Float64
    α  :: Float64
    δ  :: Float64

    Nk :: Int
    Nz :: Int

    kgrid :: CuVector{Float64}
    zgrid :: CuVector{Float64}
    Pz    :: CuMatrix{Float64}

    # Policies on fixed grid
    gk    :: CuMatrix{Float64}      # k′(k,z)
    gc    :: CuMatrix{Float64}      # c(k,z)

    # EGM work arrays (endogenous grid)
    k_endo :: CuMatrix{Float64}     # Nk × Nz, current k that makes k′ optimal
    c_endo :: CuMatrix{Float64}     # Nk × Nz, c implied by Euler
    muc    :: CuMatrix{Float64}     # Nk × Nz, E[u_c(c_{t+1}) | k′, z]
end

function GrowthCUDA(; β=0.95, σ=2.0, α=0.3, δ=0.2,
                    Nk=1000, Nz=10, kmin=0.1, kmax=10.0,
                    ρ=0.9, σϵ=0.02)

    # 1. CPU grids
    kgrid_cpu = collect(range(kmin, kmax, length = Nk))
    zgrid_cpu, Pz_cpu = tauchenlib(Nz, ρ, σϵ)
    zgrid_cpu_levels = exp.(zgrid_cpu)

    # 2. Move to GPU
    kgrid = CuArray(kgrid_cpu)
    zgrid = CuArray(zgrid_cpu_levels)
    Pz    = CuArray(Pz_cpu)

    # 3. Allocate GPU arrays
    gk     = CUDA.zeros(Float64, Nk, Nz)
    gc     = CUDA.zeros(Float64, Nk, Nz)
    k_endo = CUDA.zeros(Float64, Nk, Nz)
    c_endo = CUDA.zeros(Float64, Nk, Nz)
    muc    = CUDA.zeros(Float64, Nk, Nz)

    return GrowthCUDA(β, σ, α, δ,
                      Nk, Nz,
                      kgrid, zgrid, Pz,
                      gk, gc,
                      k_endo, c_endo, muc)
end



"""
    output(kv, zv, α)
Production function y = z * k^α.
"""
function output(kv::Float64, zv::Float64, α::Float64)
    yv = zv * kv^α
    return yv
end

@inline function muc(c::Float64, σ::Float64)
    return c^(-σ)
end

function Eval_muc!(muc, gk, kgrid, zgrid, Pz,
                   β::Float64, σ::Float64, α::Float64, δ::Float64,
                   Nk::Int, Nz::Int)

    jkp = (blockIdx().x - 1) * blockDim().x + threadIdx().x  # index for k′
    jz  = (blockIdx().y - 1) * blockDim().y + threadIdx().y  # index for z

    if jkp > Nk || jz > Nz
        return
    end

    kvp = kgrid[jkp]   # k′
    # current z index jz, but future shocks come from row jz of Pz

    Ev_term = 0.0

    @inbounds for jzp in 1:Nz
        prob = Pz[jz, jzp]
        zpv  = zgrid[jzp]

        # policy at (k′, z′): k″ = gk(k′, z′)
        kp2  = gk[jkp, jzp]

        # c_{t+1} from budget with state (k′, z′) and choice k″
        ypv  = output(kvp, zpv, α)
        cpv  = (1.0 - δ) * kvp + ypv - kp2

        if cpv > 0.0
            mucp = muc(cpv, σ)
            Rp  = 1.0 - δ + α * zpv * kvp^(α - 1.0)
            Ev_term += prob * mucp * Rp
        end
    end

    muc[jkp, jz] = Ev_term
    return
end

function muc_iter!(gw::GrowthCUDA)
    Nk, Nz = gw.Nk, gw.Nz
    threads = (16, 16)
    blocks  = (cld(Nk, threads[1]), cld(Nz, threads[2]))

    @cuda threads=threads blocks=blocks Eval_muc!(
        gw.muc, gw.gk, gw.kgrid, gw.zgrid, gw.Pz,
        gw.β, gw.σ, gw.α, gw.δ,
        Nk, Nz
    )
end

function invert_euler!(k_endo, c_endo, muc, kgrid, zgrid,
                       β::Float64, σ::Float64, α::Float64, δ::Float64,
                       Nk::Int, Nz::Int)

    jkp = (blockIdx().x - 1) * blockDim().x + threadIdx().x  # index for k′
    jz  = (blockIdx().y - 1) * blockDim().y + threadIdx().y  # index for z

    if jkp > Nk || jz > Nz
        return
    end

    kvp = kgrid[jkp]
    zv  = zgrid[jz]

    Ev_muc = muc[jkp, jz]

    # Euler inversion: current consumption
    cv = if Ev_muc <= 0.0
        1e-10
    else
        rhs = β * Ev_muc
        rhs = max(rhs, 1e-14)
        rhs^(-1.0 / σ)
    end

    # Find best current k index on existing kgrid
    jk_best = get_jk_from_cv(kgrid, zv, kvp, cv, α, δ, Nk)
    kv_star = kgrid[jk_best]

    k_endo[jkp, jz] = kv_star
    c_endo[jkp, jz] = cv

    return
end

function euler_iter!(gw::GrowthCUDA)
    Nk, Nz = gw.Nk, gw.Nz
    threads = (16, 16)
    blocks  = (cld(Nk, threads[1]), cld(Nz, threads[2]))

    @cuda threads=threads blocks=blocks invert_euler!(
        gw.k_endo, gw.c_endo, gw.muc,
        gw.kgrid, gw.zgrid,
        gw.β, gw.σ, gw.α, gw.δ,
        Nk, Nz
    )
end
function opt_policy!(gk, gc, k_endo, c_endo,
                     kgrid, zgrid,
                     α::Float64, δ::Float64,
                     Nk::Int, Nz::Int,
                     σk::Float64, τ::Float64)

    jk = (blockIdx().x - 1) * blockDim().x + threadIdx().x  # current k index
    jz = (blockIdx().y - 1) * blockDim().y + threadIdx().y  # current z index

    if jk > Nk || jz > Nz
        return
    end

    kv = kgrid[jk]
    zv = zgrid[jz]

    # 1. nearest endogenous index and neighborhood around it
    j0, idxs = get_neighborhood_indices(k_endo, kv, jz, Nk; halfwidth = 1)

    # 2. softmax weights over these neighbors based on distance in k_endo
    weights = zeros(Float64, length(idxs))
    local_softmax_from_k!(weights, k_endo, idxs, kv, jz, σk, τ)

    # 3. smoothed consumption at (k, z)
    cv = smooth_c_from_neighbors(c_endo, idxs, jz, weights)

    # 4. implied k' from budget: c = (1-δ)k + z k^α - k'
    yv  = output(kv, zv, α)
    kpv = (1.0 - δ) * kv + yv - cv

    gc[jk, jz] = cv
    gk[jk, jz] = kpv

    return
end
function policy_iter!(gw::GrowthCUDA; σk::Float64, τ::Float64 = 1.0)
    Nk, Nz = gw.Nk, gw.Nz
    threads = (16, 16)
    blocks  = (cld(Nk, threads[1]), cld(Nz, threads[2]))

    @cuda threads=threads blocks=blocks opt_policy!(
        gw.gk, gw.gc,
        gw.k_endo, gw.c_endo,
        gw.kgrid, gw.zgrid,
        gw.α, gw.δ,
        Nk, Nz,
        σk, τ
    )
end


function egm_iter!(gw::GrowthCUDA; σk::Float64, τ::Float64)
    muc_iter!(gw)
    euler_iter!(gw)
    policy_iter!(gw; σk = σk, τ = τ)
end

function egm!(gw::GrowthCUDA; max_iter=500, tol=1e-7, τ=0.01, σ_mult=2.0)
    init_policy!(gw)

    # one warmup EGM iteration to build an initial endogenous grid
    muc_iter!(gw)
    euler_iter!(gw)
    σk = compute_sigma_k(gw; multiplier = σ_mult)

    dist = Inf
    it   = 0

    while it < max_iter && dist > tol
        it += 1

        gk_old = copy(gw.gk)

        egm_iter!(gw; σk = σk, τ = τ)

        dist = maximum(abs.(gw.gk .- gk_old))

        if it % 10 == 0 || it == 1
            println("EGM iter = $it, dist = $dist")
        end
    end

    return gw
end