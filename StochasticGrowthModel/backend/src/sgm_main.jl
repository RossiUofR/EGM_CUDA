using Pkg
Pkg.activate(".")
using JLD2,CUDA,Distributions,LinearAlgebra,BenchmarkTools
include("sgm_library.jl")
include("sgm_model.jl")

gw_cuda = GrowthCUDA()
@btime egm!(gw_cuda)

gw = to_cpu(gw_cuda)

@save "StochasticGrowthModel/backend/output/sgm_egm_result.jld2" gw

