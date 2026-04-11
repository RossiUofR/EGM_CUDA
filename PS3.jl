using JLD2,CUDA,Distributions,LinearAlgebra,BenchmarkTools


#PS2 CODE
include("/home/rrossi5/Matias/ConsumptionSavings/CS_CUDA/src/backend/consumption_model.jl")
include("/home/rrossi5/Matias/ConsumptionSavings/CS_CUDA/src/backend/consumption_library.jl")
# PS3 CODE
include("/home/rrossi5/Matias/EGM_CUDA/ConsumptionSavingsEGM/backend/src/cs_model.jl")
include("/home/rrossi5/Matias/EGM_CUDA/ConsumptionSavingsEGM/backend/src/cs_library.jl")

######### CONSUMPTION MODEL TRACK ###################

# CS EGM from cs_model.jl
cs_gpu = ConsSavEGMCUDA(
                         Na = 1500, amax = 10.0,
                         Ny = 10, Nε = 5);
@btime egm!(cs_gpu;max_iter = 5000,tol = 1e-5,λ = 0.05)

# VFI from consumption_model.jl
cs_vfi_cuda = ConsSavCUDA(;method=:VFI,Na = 1500, amax = 10.0,
                         Ny = 10, Nε = 5);
@btime vfi!(cs_vfi_cuda;max_iter = 5000,tol = 1e-5)

# PFI from consumption_model.jl
cs_pfi_cuda = ConsSavCUDA(;method=:PFI,Na = 1500, amax = 10.0,
                         Ny = 10, Nε = 5);

@btime pfi!(cs_pfi_cuda;max_iter = 5000,tol = 1e-5)

################ STOCHASTIC GROWTH MODEL TRACK #####################
#PS3 CODE
include("/home/rrossi5/Matias/EGM_CUDA/StochasticGrowthModelEGM/backend/src/sgm_model.jl")
include("/home/rrossi5/Matias/EGM_CUDA/StochasticGrowthModelEGM/backend/src/sgm_library.jl")

include("/home/rrossi5/Matias/SGM/StochasticGrowthModel/StochasticGrowthModel/src/backend/growth_model.jl")
include("/home/rrossi5/Matias/SGM/StochasticGrowthModel/StochasticGrowthModel/src/backend/growth_library.jl")

gw_egm = GrowthCUDAEGM(Nk=1500, Nz=15, kmin=0.1, kmax=2.0);
@btime egm!(gw_egm;max_iter = 5000,tol = 1e-5)

gw_vfi = GrowthCUDA(Nk=1500, Nz=15, kmin=0.1, kmax=2.0,);
@btime vfi!(gw_vfi;tol = 1e-5)

@btime 
function growth_vfi()
    gw_vfi = GrowthCUDA(Nk=1500, Nz=15, kmin=0.1, kmax=2.0,);
    vfi!(gw_vfi;tol = 1e-5)
end
@btime growth_vfi()