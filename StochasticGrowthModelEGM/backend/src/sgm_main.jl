using Pkg
Pkg.activate(".")
using JLD2,CUDA,Distributions,LinearAlgebra,BenchmarkTools
include("sgm_model.jl")
include("sgm_library.jl")

gw_cuda = GrowthCUDAEGM()
egm!(gw_cuda)

gw = to_cpu(gw_cuda)

@save "sgm_egm_result.jld2" gw

