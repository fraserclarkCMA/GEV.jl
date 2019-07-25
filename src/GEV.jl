module GEV

using DataFrames, StatsModels, ForwardDiff, Parameters, LinearAlgebra, Optim

import Distributed: CachingPool, pmap, workers

VV{T} = Vector{Vector{T}}

include("Types.jl")
include("clogit.jl")
include("nlogit.jl")
include("EstimationWrappers.jl")
include("utils.jl")

export clogit, clogit_model, clogit_param, clogit_case, 
	   clogit_data, clogit_case_data, make_clogit_data, 
	   ll_clogit, ll_clogit_case, 
	   grad_clogit_case, analytic_grad_clogit_case, 
	   fg_clogit_case, analytic_fg_clogit_case, 
	   hessian_clogit_case, pmap_hessian_clogit, 
	   fgh_clogit_case, 
	   estimate_clogit,
	   nlogit, nlogit_model, nlogit_param, nlogit_case,
	   nlogit_nest_data, nlogit_data, make_nlogit_data,
	   ll_nlogit, ll_nlogit_case, 
	   grad_nlogit_case, analytic_grad_nlogit_case, 
	   fg_nlogit_case, analytic_fg_nlogit_case, 
	   hessian_nlogit_case, pmap_hessian_nlogit, 
	   fgh_nlogit_case, 	   
	   estimate_nlogit,
	   logsumexp, fun_RUM, get_vec_dict, passdata, distdata,
	   vec_to_theta!, theta_to_vec!, std_err, @formula

end 
