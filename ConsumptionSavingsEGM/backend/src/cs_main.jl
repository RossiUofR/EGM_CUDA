using Pkg
Pkg.activate(".")

using JLD2,CUDA,Distributions,LinearAlgebra,BenchmarkTools
include("cs_model.jl")
include("cs_library.jl")

cs_gpu = ConsSavEGMCUDA()
egm!(cs_gpu)
cs_egm = to_cpu(cs_gpu)

@save "cs_egm_result.jld2" cs_egm
