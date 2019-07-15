module GEV

using DataFrames, StatsModels, ForwardDiff, Parameters, LinearAlgebra

VV{T} = Vector{Vector{T}}

include("clogit.jl")
include("nlogit.jl")
include("utils.jl")

export clogit_model, clogit_param, clogit_data, clogit_case_data, 
	   clogit_loglike, make_clogit_data, clogit,
	   nlogit_model, nlogit_param, 
	   nlogit_nest_data, nlogit_data, nlogit,
	   nlogit_loglike, make_nlogit_data,
	   logsumexp, fun_sigma, fun_lambda, get_vec_dict,
	   vec_to_theta!, theta_to_vec!, std_err, @formula

end 
