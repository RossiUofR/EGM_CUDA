mutable struct ConsSavEGMCUDA
    β  :: Float64
    γ  :: Float64
    R  :: Float64
    ϕ  :: Float64
    ρ  :: Float64
    σ  :: Float64

    Na :: Int
    Ny :: Int
    Nε :: Int

    apgrid  :: CuVector{Float64}
    ygrid   :: CuVector{Float64}
    εnodes  :: CuVector{Float64}
    wε      :: CuVector{Float64}

    # EGM work arrays
    a_endo  :: CuMatrix{Float64}
    c_endo  :: CuMatrix{Float64}
    muc     :: CuMatrix{Float64}   # stores E[u_c] at (a', y)

    # Policies on fixed exogenous grid
    ga      :: CuMatrix{Float64}
    gc      :: CuMatrix{Float64}

    # Optional value function
    V       :: CuMatrix{Float64}
end

function ConsSavEGMCUDA(; β = 0.975, R = 1.02, γ = 2.0,
                         ϕ = 0.0, ρ = 0.9, σ = 0.06,
                         Na = 1500, amax = 10.0,
                         Ny = 10, Nε = 5)

    apgrid_cpu = get_log_agrid(Na, ϕ, amax)

    ylog_grid  = range(-2.0, 2.0, length = Ny)
    ygrid_cpu  = exp.(ylog_grid)

    εnodes_cpu, wε_cpu = make_quadrature(Nε)

    apgrid = CuArray(apgrid_cpu)
    ygrid  = CuArray(ygrid_cpu)
    εnodes = CuArray(εnodes_cpu)
    wε     = CuArray(wε_cpu)

    a_endo = CUDA.zeros(Float64, Na, Ny)
    c_endo = CUDA.zeros(Float64, Na, Ny)
    muc    = CUDA.zeros(Float64, Na, Ny)
    ga     = CUDA.zeros(Float64, Na, Ny)
    gc     = CUDA.zeros(Float64, Na, Ny)
    V      = CUDA.zeros(Float64, Na, Ny)

    return ConsSavEGMCUDA(β, γ, R, ϕ, ρ, σ,
                          Na, Ny, Nε,
                          apgrid, ygrid,
                          εnodes, wε,
                          a_endo, c_endo, muc,
                          ga, gc, V)
end
"""
    budget_constraint(kv, kvp, zv, α, δ)

Budget constraint in levels:

c = (1 - δ) * k + z * k^α - k'.
"""
function budget_constraint(yv::Float64, av::Float64,
                  apv::Float64, R::Float64)
    
    cv  = yv + av*R - apv
    return cv
end


"""
    u(c, σ)

CRRA utility:
- if σ = 1: u(c) = log(c)
- otherwise: u(c) = c^(1-σ) / (1-σ)

Assumes c > 0 (I will enforce this in the Bellman step).
"""
function u(c::Float64, γ::Float64)
    if γ == 1.0
        return log(c)
    else
        return c^(1.0 - γ) / (1.0 - γ)
    end
end

# marginal utility for CRRA
@inline function muc_fun(c::Float64, γ::Float64)
    return c^(-γ)
end

"""
    Eval_muc!(muc, gc, apgrid, ygrid, εnodes, wε,
              β, γ, R, ρ, σ,
              Na, Ny, Nε)

For each pair (jap, jy) ≡ (a', yv), compute

    muc[jap, jy] = E[ u_c(c_{t+1}) | a', yv ]

using the current consumption policy gc and Gaussian quadrature.
"""
function Eval_muc!(muc, gc, apgrid, ygrid, εnodes, wε,
                   β::Float64, γ::Float64, R::Float64,
                   ρ::Float64, σ::Float64,
                   Na::Int, Ny::Int, Nε::Int)

    jap = (blockIdx().x - 1) * blockDim().x + threadIdx().x  # index for a'
    jy  = (blockIdx().y - 1) * blockDim().y + threadIdx().y  # index for y today

    if jap > Na || jy > Ny
        return
    end

    apv = apgrid[jap]    # a' value (not strictly needed here)
    yv  = ygrid[jy]      # y today

    Ev_muc = 0.0
    yv_log = log(yv)

    @inbounds for jε in 1:Nε
        εv  = εnodes[jε]
        wvε = wε[jε]

        # income tomorrow
        ypv = exp(ρ * yv_log + σ * εv)

       c_star = interp_y_from_nearest(gc, jap, ypv, ygrid, Ny)

        if c_star > 0.0
            Ev_muc += wvε * muc_fun(c_star, γ)
        end
    end

    muc[jap, jy] = Ev_muc
    return
end


function muc_iter!(cs::ConsSavEGMCUDA)
    Na, Ny, Nε = cs.Na, cs.Ny, cs.Nε
    threads = (16, 16)
    blocks  = (cld(Na, threads[1]), cld(Ny, threads[2]))

    @cuda threads=threads blocks=blocks Eval_muc!(
        cs.muc, cs.gc,
        cs.apgrid, cs.ygrid,
        cs.εnodes, cs.wε,
        cs.β, cs.γ, cs.R, cs.ρ, cs.σ,
        Na, Ny, Nε
    )
end

"""
    invert_euler!(a_endo, c_endo, muc, apgrid, ygrid,
                            β, γ, R,
                            Na, Ny)

Given:
  - muc[jap, jy]  = E[u_c(c_{t+1}) | a'_j, yv_jy]
  - apgrid[jap]   = a'_j (next-period assets grid)
  - ygrid[jy]     = yv

Compute for each (jap, jy):

  1. c_endo[jap, jy] = current consumption implied by Euler equation
  2. a_endo[jap, jy] = current assets that make a' optimal today

Formulas:
  cv = (β R * muc)^(-1/γ)
  av = (c_t + a' - yv) / R
"""
function invert_euler!(a_endo, c_endo, muc, apgrid, ygrid,
                       β::Float64, γ::Float64, R::Float64,
                       Na::Int, Ny::Int)

    jap = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    jy  = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if jap > Na || jy > Ny
        return
    end

    apv = apgrid[jap]
    yv  = ygrid[jy]

    Ev_muc = muc[jap, jy]

    if Ev_muc <= 0.0
        cv = 1e-10
    else
        rhs = β * R * Ev_muc
        rhs = max(rhs, 1e-12)
        cv = rhs^(-1.0 / γ)
    end

    av = (cv + apv - yv) / R


    c_endo[jap, jy] = cv
    a_endo[jap, jy] = av

    return
end
"""
    euler_iter(cs)

Use cs.muc to fill cs.c_endo and cs.a_endo via Euler inversion.
"""
function euler_iter!(cs::ConsSavEGMCUDA)
    Na, Ny = cs.Na, cs.Ny
    threads = (16, 16)
    blocks  = (cld(Na, threads[1]), cld(Ny, threads[2]))

    @cuda threads=threads blocks=blocks invert_euler!(
        cs.a_endo, cs.c_endo, cs.muc,
        cs.apgrid, cs.ygrid,
        cs.β, cs.γ, cs.R,
        Na, Ny
    )
end

"""
    opt_policy!(ga, gc, a_endo, c_endo,
                           apgrid, ygrid,
                           R, Na, Ny)

For each fixed current state (ja, jy) ≡ (av, yv) on the exogenous grid:

  1. Search over jap = 1..Na to find the endogenous asset
     a_endo[jap, jy] closest to av.
  2. Take c = c_endo[jap_best, jy].
  3. Set a' = R*av + yv - c.

Store:
  gc[ja, jy] = c(a,y)
  ga[ja, jy] = a'(a,y)
"""
function opt_policy!(ga, gc, a_endo, c_endo, apgrid, ygrid,
                     R::Float64, Na::Int, Ny::Int)

    ja = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    jy = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if ja > Na || jy > Ny
        return
    end

    av = apgrid[ja]
    yv = ygrid[jy]

    # endogenous a-grid threshold for this y
    a1 = a_endo[1, jy]   # lowest current-asset in endogenous grid at this y

    if av <= a1
        # ---- borrowing constraint region: a' = 0 when ϕ = 0 ----
        apv = 0.0
        cv  = R * av + yv - apv   # c + a' = Ra + y
    else
        # ---- interior region: use EGM inversion as usual ----
        jap_star = get_jap(a_endo, av, jy, Na)
        cv     = interp_c_in_a(a_endo, c_endo, av, jy, Na, jap_star)
        apv    = R * av + yv - cv
    end

    # final safety clamp (handles tiny negatives)
    if apv < 0.0
        apv = 0.0
        cv  = R * av + yv - apv
    end

    gc[ja, jy] = cv
    ga[ja, jy] = apv

    return
end
"""
    project_policy!(cs)

Fill cs.gc and cs.ga on the fixed grid (apgrid, ygrid)
using nearest neighbor on (a_endo, c_endo).
"""
function policy_iter!(cs::ConsSavEGMCUDA)
    Na, Ny = cs.Na, cs.Ny
    threads = (16, 16)
    blocks  = (cld(Na, threads[1]), cld(Ny, threads[2]))

    @cuda threads=threads blocks=blocks opt_policy!(
        cs.ga, cs.gc,
        cs.a_endo, cs.c_endo,
        cs.apgrid, cs.ygrid,
        cs.R,
        Na, Ny
    )
end

function egm_iter!(cs::ConsSavEGMCUDA)
    # 1. Expected marginal utility on (a', yv)
    muc_iter!(cs)

    # 2. Euler inversion → endogenous grid
    euler_iter!(cs)

    # 3. Project back to fixed grid using nearest neighbor
    policy_iter!(cs)
end









"""
    init_policy!(ga, gc, apgrid, ygrid, ϕ, R, Na, Ny)

GPU kernel:
  For each (ja, jy), set

      ga[ja,jy] = -ϕ
      gc[ja,jy] = yv + R * av + ϕ

which is a simple feasible starting policy.
"""
function fill_guess!(ga, gc, apgrid, ygrid,
                             ϕ::Float64, R::Float64,
                             Na::Int, Ny::Int)

    ja = (blockIdx().x - 1) * blockDim().x + threadIdx().x  # index for a
    jy = (blockIdx().y - 1) * blockDim().y + threadIdx().y  # index for y

    if ja > Na || jy > Ny
        return
    end

    av = apgrid[ja]
    yv = ygrid[jy]

    apv = -ϕ
    cv  = yv + R * av - apv   # = yv + R*av + ϕ

    ga[ja, jy] = apv
    gc[ja, jy] = cv

    return
end

"""
    init_policy!(cs)

GPU-native initialization of the policy guess for EGM.
"""
function init_policy!(cs::ConsSavEGMCUDA)
    Na, Ny = cs.Na, cs.Ny
    threads = (16, 16)
    blocks  = (cld(Na, threads[1]), cld(Ny, threads[2]))

    @cuda threads=threads blocks=blocks fill_guess!(
        cs.ga, cs.gc,
        cs.apgrid, cs.ygrid,
        cs.ϕ, cs.R,
        Na, Ny
    )

    return
end




"""
    egm!(cs; max_iter = 500, tol = 1e-6)

Solve the consumption-savings problem by EGM.

Convergence is checked using the sup norm on the asset policy:
    max |ga_new - ga_old|
"""
function egm!(cs::ConsSavEGMCUDA; max_iter = 15000, tol = 1e-10, λ = 0.05)
    
    init_policy!(cs)
    diff = Inf
    jt   = 0

    while jt < max_iter && diff > tol
        jt += 1
        # store old policies
        ga_old = copy(cs.ga)
        gc_old = copy(cs.gc)

        # one raw EGM update
        egm_iter!(cs)

        # raw new policies produced by egm_iter!
        ga_new = copy(cs.ga)
        gc_new = copy(cs.gc)
                # convergence metric on asset policy

        diff = maximum(abs.(cs.ga .- ga_old))

        # damped update:
        # new iterate = λ * raw_update + (1-λ) * old_iterate
        cs.ga .= λ .* ga_new .+ (1.0 - λ) .* ga_old
        cs.gc .= λ .* gc_new .+ (1.0 - λ) .* gc_old

        if jt % 10 == 0 || jt == 1
            println("EGM iter = ", jt, ", dist = ", diff)
        end
    end

    return cs
end