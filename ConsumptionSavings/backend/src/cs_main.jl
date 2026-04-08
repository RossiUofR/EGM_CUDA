using Pkg
Pkg.activate(".")

using JLD2,CUDA,Distributions,LinearAlgebra,BenchmarkTools
include("cs_library.jl")
include("cs_model.jl")

cs_gpu = ConsSavEGMCUDA()
@btime egm!(cs_gpu)
cs_egm = to_cpu(cs_gpu)

cd("ConsumptionSavings/backend/output")
@save "cs_egm_result.jld2" cs_egm
