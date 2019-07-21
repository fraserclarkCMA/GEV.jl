# Two-level nested logit

@with_kw struct nlogit_param
	beta 	:: Vector{Real} 	# Parameters varying within an nest
	alpha 	:: Vector{Real}		# Parameters varying across nests 
	tau 	:: Vector{Real} 	# Nesting Parameters pre-logit transform
	lambda 	:: Vector{Real}		# Nesting Parameters post-transform
end

# Constructor functions for nlogit_param
#nlogit_param() = nlogit_param(Vector{Float64}(),Vector{Float64}(),Vector{Float64}(),Vector{Float64}())
nlogit_param(NX::Int64,NW::Int64,NN::Int64) where T<:Real = nlogit_param(zeros(Float64,NX),zeros(Float64,NW),zeros(Float64,NN),zeros(Float64,NN))

@with_kw struct nlogit_model
	f_beta 	:: StatsModels.FormulaTerm 	# StatsModels.FormulaTerm for variable that vary within nests 
	f_alpha :: StatsModels.FormulaTerm	# StatsModels.FormulaTerm for variables that vary across nests (f_alpha = @formula(d ~ Z&nestid) or not identifed (i.e. Z might be HH chars))
	params 	:: nlogit_param 	# Parameters
	case_id	:: Symbol 			# 1 if option chosen, 0 if not chosen
	nest_id :: Symbol 			# Identifier for nest identifier variable
	choice_id :: Symbol 		# Choice identifier variable
	idx 	:: Dict 			# Index for parameters
	nx 		:: Int64 			# Number of parameters 
	coefnames :: Vector 		# Coef names
	opts 	:: Dict 			# Group id var, number of different nesting parameters, nest id
	flags	:: Dict 			# Flags indicating which of beta, alpha and tau are estimated	
end

# Constructor functions for nlogit_model

# Multiple dispatch to omit nest specific variable (i.e. logit + nest parameters)
function nlogit_model(f_beta::StatsModels.FormulaTerm, df::DataFrame; case::Symbol, nests::Symbol, choice_id::Symbol, RUM::Bool=true, num_lambda::Int64=1, nest_labels=levels(df[nests]))
	f_alpha = @eval(@formula($(f_beta.lhs.sym) ~ 0))

	f_beta = StatsModels.apply_schema(f_beta, StatsModels.schema(f_beta, df))
	f_alpha = StatsModels.apply_schema(f_alpha, StatsModels.schema(f_alpha, df))
	NX = isa(f_beta.rhs.terms[1], StatsModels.InterceptTerm{false}) ? 0 : size(modelcols(f_beta , df)[2], 2)
	NW = isa(f_alpha.rhs.terms[1], StatsModels.InterceptTerm{false}) ? 0 : size(modelcols(f_alpha , df)[2], 2) 
	np = nlogit_param(NX, NW, num_lambda )
	opts = Dict(:RUM=>RUM, :num_lambda=>num_lambda)
	flags = Dict(:beta=>true, :alpha=>false, :tau=>true)
	idx, nx = get_vec_dict(np, flags)
	coefnms = num_lambda>1 ? 
		vcat(coefnames(f_beta)[2],["tau_$(i)" for i in nest_labels]) :
		vcat(coefnames(f_beta)[2],["tau"])
	return nlogit_model(f_beta, f_alpha, np, case, nests, choice_id, idx, nx, coefnms, opts, flags)
end

# Multiple dispatch to include nest specific variable (i.e. logit + nest parameters)
function nlogit_model(f_beta::StatsModels.FormulaTerm, f_alpha::StatsModels.FormulaTerm, df::DataFrame; case::Symbol, nests::Symbol, choice_id::Symbol, RUM::Bool=true, num_lambda::Int64=1, nest_labels=levels(df[nests])) 
	f_beta = StatsModels.apply_schema(f_beta, StatsModels.schema(f_beta, df))
	f_alpha = StatsModels.apply_schema(f_alpha, StatsModels.schema(f_alpha, df))
	NX = isa(f_beta.rhs.terms[1], StatsModels.InterceptTerm{false}) ? 0 : size(modelcols(f_beta , df)[2], 2)
	NW = isa(f_alpha.rhs.terms[1], StatsModels.InterceptTerm{false}) ? 0 : size(modelcols(f_alpha , df)[2], 2) 
	np = nlogit_param(NX, NW, num_lambda)
	opts = Dict(:RUM=>RUM, :num_lambda=>num_lambda)
	flags = Dict(:beta=>true, :alpha=>true, :tau=>true)
	idx, nx = get_vec_dict(np, flags)
	coefnms = num_lambda>1 ? 
		vcat(coefnames(f_beta)[2],coefnames(f_alpha)[2],["tau_$(i)" for i in nest_labels]) :
		vcat(coefnames(f_beta)[2],coefnames(f_alpha)[2],["tau"])
	return nlogit_model(f_beta, f_alpha, np, case, nests, choice_id, idx, nx, coefnms, opts, flags)
end

struct nlogit_nest_data
	jstar 		:: Int64 				# Position of chosen option in chosen nest in the data
	dstar 		:: Union{Int64,String}	# Identifier of chosen option 
	nest_star	:: Int64 				# Chosen nest 
	nest_num 	:: Int64
	Xj			:: Matrix{Float64}		# Matrix of regressors for within-nest
	Wk 			:: Matrix{Float64}		# Matrix of regressors for across-nest
end
nlogit_nest_data() = new()

nlogit_data = VV{nlogit_nest_data}

# Construct the model data set 
function make_nlogit_data(model::nlogit_model, df::DataFrame)
	@unpack f_beta, f_alpha, params, case_id, nest_id, choice_id, opts, flags = model 
	f_beta = StatsModels.apply_schema(f_beta, StatsModels.schema(f_beta, df))
	f_alpha = StatsModels.apply_schema(f_alpha, StatsModels.schema(f_alpha, df))
	dataset = Vector{nlogit_nest_data}[]
	for casedf in groupby(df, case_id)
		casedata = []
		j_idx = findall(x->x==1, casedf[f_beta.lhs.sym])
		dstar = casedf[j_idx, choice_id][1]
		nest_star = casedf[j_idx, nest_id][1]
		for nestdf in groupby(casedf, nest_id)
			nestnum = nestdf[1,:nestnum]
			if nestnum == nest_star
				jstar = findall(x->x==1, nestdf[f_beta.lhs.sym])[1]
			else
				jstar = 0
			end
			Xj = isa(f_beta.rhs.terms[1], StatsModels.InterceptTerm{false}) ?  Matrix{Float64}(undef, 0, 0) : modelcols(f_beta , nestdf)[2]
			Wk = isa(f_alpha.rhs.terms[1], StatsModels.InterceptTerm{false}) ? Matrix{Float64}(undef, 0, 0) : modelcols(f_alpha , nestdf)[2] 
			push!(casedata, nlogit_nest_data(jstar, dstar, nest_star, nestnum, Xj, Wk ))
		end
		push!(dataset, casedata)
	end
	return dataset
end

struct nlogit 
	model 	:: nlogit_model
	data 	:: nlogit_data
end

# Consider logit transform for lambda
fun_sigma(τ::T) where T<:Real = 1.0 .+ exp.(-τ)
fun_lambda(τ::T) where T<:Real = 1.0 ./ (1.0 .+ exp.(-τ))
fun_sigma(τ::Vector{T}) where T<:Real = 1.0 .+ exp.(-τ)
fun_lambda(τ::Vector{T}) where T<:Real = 1.0 ./ (1.0 .+ exp.(-τ))

# Specify using dictionary of flags which sets of parameters are being estimated -> location of each type of parameter in x (input to optimiser)
function get_vec_dict(θ::nlogit_param, flags::Dict)
	@unpack beta, alpha, tau, lambda = θ
	idx = Dict()
	x = Float64[]
	if flags[:beta]
		append!(x, beta)
		idx[:beta] = 1:length(x)
	end
	if flags[:alpha]
		append!(x, alpha)
		idx[:alpha] = length(x)-length(alpha)+1:length(x)
	end
	if flags[:tau]
		append!(x, tau)
		idx[:tau] = length(x)-length(tau)+1:length(x)
	end
	return idx, length(x)
end
get_vec_dict(nlm::nlogit_model) = get_vec_dict(nlm.params, nlm.flags)

# Copy x (input to optimiser) into nlogit type
function vec_to_theta!(x::Vector{T}, θ::nlogit_param, flags::Dict, idx::Dict) where T<:Real
	if flags[:beta]
		θ.beta[:] = x[idx[:beta]]
	end
	if flags[:alpha]
		θ.alpha[:] = x[idx[:alpha]]
	end
	if flags[:tau]
		θ.tau[:] = x[idx[:tau]]
		#copyt0!(θ.lambda, fun_lambda(θ.tau))
	end
end

#  Copy nlogit parameter type into x (input to optimiser)
function theta_to_vec!(x::Vector{T}, θ::nlogit_param, flags::Dict, idx::Dict) where T<:Real
	@unpack beta, alpha, tau, lambda = θ
	if flags[:beta]
		copyto!(x[idx[:beta]], beta)
	end
	if flags[:alpha]
		copyto!(x[idx[:alpha]], alpha )
	end
	if flags[:tau]
		copyto!(x[idx[:tau]], tau )
	end
end

function ll_nlogit(x::Vector{T}, θ::nlogit_param, nld::nlogit_data, flags::Dict, idx::Dict, RUM::Bool) where T<:Real
	vec_to_theta!(x, θ, flags, idx)	
	@unpack beta, alpha, tau, lambda = θ
	
	Numλ = length(tau)	

	LL = 0.
	for case_data in nld 
		denom = Vector{Real}()
		for nest_data in case_data
			@unpack jstar, dstar, nest_star, nest_num, Xj, Wk = nest_data
			if Numλ > 1 
				λ_k = RUM ? fun_lambda(tau[nest_num]) : tau[nest_num]
				σ_k = RUM ? fun_sigma(tau[nest_num]) : 1.0 / tau[nest_num]
			else 
				λ_k = RUM ? fun_lambda(tau[1]) : tau[1]
				σ_k = RUM ? fun_sigma(tau[1]) : 1.0 / tau[1]
			end
			V = Xj*beta /λ_k
			if flags[:alpha]
				W = Wk*alpha
				if nest_star == nest_num
					LL += V[jstar] + W[1] + (λ_k - 1.0)*logsumexp(V)
				end
				push!(denom, W[1] + λ_k*logsumexp(V))	
			else 
				if nest_star == nest_num
					LL += V[jstar] + (λ_k - 1.0)*logsumexp(V)
				end
				push!(denom, λ_k*logsumexp(V))	
			end		
		end
		LL += -logsumexp(denom)
	end
	return -LL
end

# loglike wrapper for nlogit_model
ll_nlogit(x::Vector{T}, nlmod::nlogit_model, nldata::nlogit_data) where T<:Real = 
	ll_nlogit(x, nlmod.params, nldata, nlmod.flags, nlmod.idx, nlmod.opts[:RUM])

ll_nlogit(x::Vector{T}, nl::nlogit) where T<:Real = ll_nlogit(x, nl.model, nl.data)

# Functions to wrap in parallel version
function ll_nlogit_case(x::Vector{T}, θ::nlogit_param, nld::nlogit_data, id::Int64, 
										flags::Dict, idx::Dict, RUM::Bool) where T<:Real 
	vec_to_theta!(x, θ, flags, idx)	
	@unpack beta, alpha, tau, lambda = θ
	Numλ = length(tau)	
	LL = 0.
	denom = Vector{Real}()
	for nest_data in nld[id]
		@unpack jstar, dstar, nest_star, nest_num, Xj, Wk = nest_data
		if Numλ > 1 
			λ_k = RUM ? fun_lambda(tau[nest_num]) : tau[nest_num]
			σ_k = RUM ? fun_sigma(tau[nest_num]) : 1.0 / tau[nest_num]
		else 
			λ_k = RUM ? fun_lambda(tau[1]) : tau[1]
			σ_k = RUM ? fun_sigma(tau[1]) : 1.0 / tau[1]
		end
		V = Xj*beta /λ_k
		if flags[:alpha]
			W = Wk*alpha
			if nest_star == nest_num
				LL += V[jstar] + W[1] + (λ_k - 1.0)*logsumexp(V)
			end
			push!(denom, W[1] + λ_k*logsumexp(V))	
		else 
			if nest_star == nest_num
				LL += V[jstar] + (λ_k - 1.0)*logsumexp(V)
			end
			push!(denom, λ_k*logsumexp(V))	
		end		
	end
	LL += -logsumexp(denom)
	return -LL 
end

ll_nlogit_case(x::Vector{T},  nlmod::nlogit_model, nldata::nlogit_data, id::Int64) where T<:Real = 
	ll_nlogit_case(x, nlmod.params, nldata, id, nlmod.flags, nlmod.idx, nlmod.opts[:RUM])

ll_nlogit_case(x::Vector{T}, nl::nlogit, id::Int64) where T<:Real = ll_nlogit_case(x, nl.model, nl.data, id)

struct nlogit_case{T<:Real} 
	F :: T
	G :: Vector{T}
	H :: Matrix{T}
end

function grad_nlogit_case(theta::Vector{T}, nl::nlogit, id::Int64) where T<:Real
	ForwardDiff.gradient(x->ll_nlogit_case(x, nl, id), theta)
end

function hessian_nlogit_case(theta::Vector{T}, nl::nlogit, id::Int64) where T<:Real
	ForwardDiff.hessian(x->ll_nlogit_case(x, nl, id), theta)
end

function fg_nlogit_case(theta::Vector{T}, nl::nlogit, id::Int64) where T<:Real
	nlogit_case(ll_nlogit_case(theta, nl, id), grad_nlogit_case(theta, nl, id), Matrix{T}(undef,0,0)) 
end

function fgh_nlogit_case(theta::Vector{T}, nl::nlogit, id::Int64) where T<:Real
	nlogit_case(ll_nlogit_case(theta, nl, id), grad_nlogit_case(theta, nl, id), hessian_nlogit_case(theta, nl, id)) 
end


function estimate_nlogit(nl::nlogit; opt_mode = :serial, opt_method = :none, 
						x_initial = randn(nl.model.nx), algorithm = LBFGS(), 
						optim_opts = Optim.Options(), workers=workers())


	clos_ll_nlogit_case(pd) = ll_nlogit_case(pd.theta, nl, pd.id)
	clos_grad_nlogit_case(pd) = ll_nlogit_case(pd.theta, nl, pd.id)
	clos_hessian_nlogit_case(pd) = hessian_nlogit_case(pd.theta, nl, pd.id)
	clos_fg_nlogit_case(pd) = fg_nlogit_case(pd.theta, nl, pd.id)
	clos_fgh_nlogit_case(pd) = fgh_nlogit_case(pd.theta, nl, pd.id)
	
	if opt_mode == :parallel
		pool = CachingPool(workers)

		function pmap_nlogit_ll(theta::Vector{T}) where T<:Real 
			pd = [passdata(theta, i) for i in 1:length(nl.data)]	
			sum(pmap(clos_ll_nlogit_case, pool, pd))
		end

		function pmap_nlogit_grad(theta::Vector{T}) where T<:Real
			pd = [passdata(theta, i) for i in 1:length(nl.data)]	
			sum(pmap(clos_grad_clogit_case, pool, pd))
		end

		function pmap_nlogit_Hess(theta::Vector{T}) where T<:Real
			pd = [passdata(theta, i) for i in 1:length(nl.data)]	
			sum(pmap(clos_hessian_nlogit_case, pool, pd))
		end

		function pmap_nlogit_fg!(F, G, theta::Vector{T}) where T<:Real
			pd = [passdata(theta, i) for i in 1:length(nl.data)]	
			nlc = pmap(clos_fg_nlogit_case, pool, pd)
			if G != nothing
				G[:] = sum([y.G for y in nlc])
			end
			if F != nothing
				return sum([y.F for y in nlc])
			end
		end

		function pmap_nlogit_fgh!(F, G, H, theta::Vector{T}) where T<:Real
			pd = [passdata(theta, i) for i in 1:length(nl.data)]	
			nlc = pmap(clos_fgh_nlogit_case, pool, pd)
			if H != nothing
				H[:] = sum([y.H for y in nlc])
			end
			if G != nothing
				G[:] = sum([y.G for y in nlc])
			end
			if F != nothing
				return sum([y.F for y in nlc])
			end
		end

		if opt_method == :grad
			out = Optim.optimize(Optim.only_fg!(pmap_nlogit_fg!), x_initial, algorithm, optim_opts)
		elseif opt_method == :hess
			out = Optim.optimize(Optim.only_fgh!(pmap_nlogit_fgh!), x_initial, algorithm, optim_opts)
		else 
			out = Optim.optimize(pmap_nlogit_ll, x_initial, NelderMead(), optim_opts)
		end

	elseif opt_mode == :serial 
		function map_nlogit_ll(theta::Vector{T}) where T<:Real 
			pd = [passdata(theta, i) for i in 1:length(nl.data)]	
			sum(map(clos_ll_nlogit_case, pd))
		end

		function map_nlogit_grad(theta::Vector{T}) where T<:Real
			pd = [passdata(theta, i) for i in 1:length(nl.data)]	
			sum(map(clos_grad_clogit_case,  pd))
		end

		function map_nlogit_Hess(theta::Vector{T}) where T<:Real
			pd = [passdata(theta, i) for i in 1:length(nl.data)]	
			sum(map(clos_hessian_nlogit_case,  pd))
		end

		function map_nlogit_fg!(F, G, theta::Vector{T}) where T<:Real
			pd = [passdata(theta, i) for i in 1:length(nl.data)]	
			nlc = map(clos_fg_nlogit_case, pd)
			if G != nothing
				G[:] = sum([y.G for y in nlc])
			end
			if F != nothing
				return sum([y.F for y in nlc])
			end
		end

		function map_nlogit_fgh!(F, G, H, theta::Vector{T}) where T<:Real
			pd = [passdata(theta, i) for i in 1:length(nl.data)]	
			nlc = map(clos_fgh_nlogit_case, pd)
			if H != nothing
				H[:] = sum([y.H for y in nlc])
			end
			if G != nothing
				G[:] = sum([y.G for y in nlc])
			end
			if F != nothing
				return sum([y.F for y in nlc])
			end
		end

		if opt_method == :grad
			out = Optim.optimize(Optim.only_fg!(map_nlogit_fg!), x_initial, algorithm, optim_opts)
		elseif opt_method == :hess
			out = Optim.optimize(Optim.only_fgh!(map_nlogit_fgh!), x_initial, algorithm, optim_opts)
		else 
			out = Optim.optimize(map_nlogit_ll, x_initial, NelderMead(), optim_opts)
		end

	else
		out = Optim.optimize(x->ll_nlogit(x, nl.data), x_initial, algorithm, optim_opts; autodiff=:forward )
	end 

	return out
end
