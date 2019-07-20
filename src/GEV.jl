module GEV

using DataFrames, StatsModels, ForwardDiff, Parameters, LinearAlgebra, Optim

import Distributed: CachingPool, pmap, workers

VV{T} = Vector{Vector{T}}

include("clogit.jl")
include("nlogit.jl")
include("utils.jl")

export clogit, clogit_model, clogit_param, clogit_data, clogit_case_data, make_clogit_data, 
	   ll_clogit, ll_clogit_case, grad_clogit_case, hessian_clogit_case, 
	   fg_clogit_case, fgh_clogit_case, estimate_clogit,
	   nlogit_model, nlogit_param, nlogit_nest_data, nlogit_data, nlogit,
	   ll_nlogit, nlogit_case_data, make_nlogit_data, dist_ll_nlogit, estimate_nlogit,
	   logsumexp, fun_sigma, fun_lambda, get_vec_dict, passdata,
	   vec_to_theta!, theta_to_vec!, std_err, @formula

end 
