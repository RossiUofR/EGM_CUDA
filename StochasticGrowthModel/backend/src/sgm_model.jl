mutable struct GrowthCUDA
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

function GrowthCUDA(; β=0.95, σ=1.0, α=0.3, δ=0.2,
                    Nk=1500, Nz=15, kmin=0.1, kmax=2.0,
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

@inline function muc_fun(c::Float64, σ::Float64)
    if abs(σ - 1.0) < 1e-12
        return 1.0 / c
    else
        return c^(-σ)
    end
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
        kppv  = gk[jkp, jzp]

        # c_{t+1} from budget with state (k′, z′) and choice k″
        ypv  = output(kvp, zpv, α)
        cpv  = (1.0 - δ) * kvp + ypv - kppv

        if cpv > 0.0
            mucp = muc_fun(cpv, σ)
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

    jkp = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    jz  = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if jkp > Nk || jz > Nz
        return
    end

    kvp = kgrid[jkp]
    zv  = zgrid[jz]

    Ev_muc = muc[jkp, jz]

    # Euler inversion
    cv = if Ev_muc <= 0.0
        1e-10
    else
        rhs = β * Ev_muc
        rhs = max(rhs, 1e-14)
        rhs^(-1.0 / σ)
    end

    # endogenous current k from budget equation
    kv_star = get_k_from_cv(kgrid, zv, kvp, cv, α, δ, Nk)

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
                     Nk::Int, Nz::Int)

    jk = (blockIdx().x - 1) * blockDim().x + threadIdx().x  # current k index
    jz = (blockIdx().y - 1) * blockDim().y + threadIdx().y  # current z index

    if jk > Nk || jz > Nz
        return
    end

    kv = kgrid[jk]
    zv = zgrid[jz]

    # nearest endogenous index for this (k,z)
    jkp_star = get_jkp(k_endo, kv, jz, Nk)

    # consumption at nearest endogenous point
    cv = c_endo[jkp_star, jz]

    # implied k' from budget: c = (1-δ)k + z k^α - k'
    yv  = output(kv, zv, α)
    kpv = (1.0 - δ) * kv + yv - cv

    gc[jk, jz] = cv
    gk[jk, jz] = kpv

    return
end



function policy_iter!(gw::GrowthCUDA)
    Nk, Nz = gw.Nk, gw.Nz
    threads = (16, 16)
    blocks  = (cld(Nk, threads[1]), cld(Nz, threads[2]))

    @cuda threads=threads blocks=blocks opt_policy!(
        gw.gk, gw.gc,
        gw.k_endo, gw.c_endo,
        gw.kgrid, gw.zgrid,
        gw.α, gw.δ,
        Nk, Nz
    )
end


function egm_iter!(gw::GrowthCUDA)
    muc_iter!(gw)
    euler_iter!(gw)
    policy_iter!(gw)
end

function egm!(gw::GrowthCUDA; max_iter=10000, tol=1e-7, λ=0.05)
    init_policy!(gw)

    dist = Inf
    jt   = 0

    while jt < max_iter && dist > tol
        jt += 1

        # store old policies
        gk_old = copy(gw.gk)
        gc_old = copy(gw.gc)

        # one raw EGM update (overwrites gw.gk, gw.gc)
        egm_iter!(gw)

        # raw new policies
        gk_raw = copy(gw.gk)
        gc_raw = copy(gw.gc)

        # damped update
        gw.gk .= (1.0 - λ) .* gk_old .+ λ .* gk_raw
        gw.gc .= (1.0 - λ) .* gc_old .+ λ .* gc_raw

        # convergence metric on capital policy
        dist = maximum(abs.(gw.gk .- gk_old))

        if jt % 30 == 0 || jt == 1
            println("EGM iter = $jt, dist = $dist")
        end
    end

    return gw
end