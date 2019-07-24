# Two-level nested logit

@with_kw struct nlogit_param
	beta 	:: Vector{Real} 	# Parameters varying within an nest
	alpha 	:: Vector{Real}		# Parameters varying across nests 
	lambda 	:: Vector{Real}		# Nesting Parameters post-transform
end

# Constructor functions for nlogit_param
#nlogit_param() = nlogit_param(Vector{Float64}(),Vector{Float64}(),Vector{Float64}(),Vector{Float64}())
nlogit_param(NX::Int64,NW::Int64,NN::Int64) where T<:Real = nlogit_param(zeros(Float64,NX),zeros(Float64,NW),zeros(Float64,NN))

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
	flags = Dict(:beta=>true, :alpha=>false, :lambda=>true)
	idx, nx = get_vec_dict(np, flags)
	coefnms = num_lambda>1 ? 
		vcat(coefnames(f_beta)[2],["λ_$(i)" for i in nest_labels]) :
		vcat(coefnames(f_beta)[2],["λ"])
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
	flags = Dict(:beta=>true, :alpha=>true, :lambda=>true)
	idx, nx = get_vec_dict(np, flags)
	coefnms = num_lambda>1 ? 
		vcat(coefnames(f_beta)[2],coefnames(f_alpha)[2],["λ_$(i)" for i in nest_labels]) :
		vcat(coefnames(f_beta)[2],coefnames(f_alpha)[2],["λ"])
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
fun_RUM(τ::T) where T<:Real = 1.0 ./ (1.0 .+ exp.(-τ))
fun_RUM(τ::Vector{T}) where T<:Real = 1.0 ./ (1.0 .+ exp.(-τ))

# Specify using dictionary of flags which sets of parameters are being estimated -> location of each type of parameter in x (input to optimiser)
function get_vec_dict(θ::nlogit_param, flags::Dict)
	@unpack beta, alpha, lambda = θ
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
	if flags[:lambda]
		append!(x, lambda)
		idx[:lambda] = length(x)-length(lambda)+1:length(x)
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
	if flags[:lambda]
		θ.lambda[:] = x[idx[:lambda]]
	end
end

#  Copy nlogit parameter type into x (input to optimiser)
function theta_to_vec!(x::Vector{T}, θ::nlogit_param, flags::Dict, idx::Dict) where T<:Real
	@unpack beta, alpha, lambda = θ
	if flags[:beta]
		copyto!(x[idx[:beta]], beta)
	end
	if flags[:alpha]
		copyto!(x[idx[:alpha]], alpha )
	end
	if flags[:lambda]
		copyto!(x[idx[:lambda]], lambda )
	end
end

function ll_nlogit(x::Vector{T}, θ::nlogit_param, nld::nlogit_data, flags::Dict, idx::Dict, RUM::Bool) where T<:Real
	vec_to_theta!(x, θ, flags, idx)	
	@unpack beta, alpha, lambda = θ
	
	Numλ = length(lambda)

	LL = 0.
	for case_data in nld 
		denom = Vector{Real}()
		for nest_data in case_data
			@unpack jstar, dstar, nest_star, nest_num, Xj, Wk = nest_data
			if Numλ > 1 
				λ_k = RUM ? fun_RUM(lambda[nest_num]) : lambda[nest_num]
			else 
				λ_k = RUM ? fun_RUM(lambda[1]) : lambda[1]
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
	@unpack beta, alpha, lambda = θ
	Numλ = length(lambda)	
	LL = 0.
	D = Vector{Real}()
	for nest_data in nld[id]
		@unpack jstar, dstar, nest_star, nest_num, Xj, Wk = nest_data
		if Numλ > 1 
			λ_k = RUM ? fun_RUM(lambda[nest_num]) : lambda[nest_num]
		else 
			λ_k = RUM ? fun_RUM(lambda[1]) : lambda[1]
		end
		V = Xj*beta /λ_k
		IV = logsumexp(V)
		if flags[:alpha]
			W = sum(Wk[1,:].*alpha)
			if nest_star == nest_num
				LL += V[jstar] + W + (λ_k - 1.0)*IV
			end
			push!(D, W + λ_k*IV)	
		else 
			if nest_star == nest_num
				LL += V[jstar] + (λ_k - 1.0)*IV
			end
			push!(D, λ_k*IV)	
		end		
	end
	LL -= logsumexp(D)
	return -LL 
end

ll_nlogit_case(x::Vector{T},  model::nlogit_model, data::nlogit_data, id::Int64) where T<:Real = 
	ll_nlogit_case(x, model.params, data, id, model.flags, model.idx, model.opts[:RUM])

ll_nlogit_case(x::Vector{T}, nl::nlogit, id::Int64) where T<:Real = ll_nlogit_case(x, nl.model, nl.data, id)

struct nlogit_case{T<:Real} 
	F :: T
	G :: Vector{T}
	H :: Matrix{T}
end

function grad_nlogit_case(theta::Vector{T}, model::nlogit_model, data::nlogit_data, id::Int64) where T<:Real
	ForwardDiff.gradient(x->ll_nlogit_case(x, model, data, id), theta)
end


function multinomial(V::Vector{T}) where T<:Real
	maxV = maximum(V)
	return  exp.(V .- maxV) ./sum(exp.(V .- maxV))
end

function analytic_grad_nlogit_case(x::Vector{Float64}, θ::nlogit_param, nld::nlogit_data, id::Int64, 
										flags::Dict, idx::Dict, RUM::Bool)  
	vec_to_theta!(x, θ, flags, idx)	
	@unpack beta, alpha, lambda = θ

	Nbeta = length(beta)
	Nalpha = length(alpha)
	Numλ = length(lambda)	

	## Allocate memory
	grad = zeros(Nbeta + Nalpha + Numλ)

	D = Vector{Float64}()
	if Nbeta>1 
		∂D_∂β = VV{Float64}()
	else 
		∂D_∂β = Vector{Float64}()
	end 
	if flags[:alpha] & Nalpha>1 
		∂D_∂α =VM{Float64}()
	else 
		∂D_∂α = VV{Float64}()
	end 
	∂D_∂λ = Vector{Float64}()
	nest_idx = Int64[]

	## Compute gradient loop over nest first, then construct across nest components of ll afterwards
	for nest_data in nld[id]

		@unpack jstar, dstar, nest_star, nest_num, Xj, Wk = nest_data

		J,K = size(Xj)

		if Numλ > 1 
			λ_k = RUM ? fun_lambda(lambda[nest_num]) : lambda[nest_num]
		else 
			λ_k = RUM ? fun_lambda(lambda[1]) : lambda[1]
		end
			
		V = Xj*beta /λ_k 	# Jl by 1
		∂V_∂β = Xj ./λ_k 	# J x K of ∂Vj_∂βk = ∂V_∂β[j,k]
		∂V_∂λ = -V ./λ_k 	# Jl by 1 

		IV = logsumexp(V)  	# Scalar
		∂IV_∂V = multinomial(V)  # Jl by 1
		∂IV_∂β = sum(repeat(∂IV_∂V, 1, K).*∂V_∂β,dims=1)[:]
		∂IV_∂λ = sum(∂IV_∂V.*∂V_∂λ, dims=1)[1]

		push!(nest_idx, nest_num)

		if nest_num == nest_star
			grad[idx[:beta]] .+= view(∂V_∂β,jstar,:) .+ (λ_k - 1.0).*∂IV_∂β 
			if Numλ > 1 
				grad[idx[:lambda][nest_num]] += ∂V_∂λ[jstar] + (λ_k - 1.0)*∂IV_∂λ + IV 
			else 
				grad[idx[:lambda][1]] += ∂V_∂λ[jstar] + (λ_k - 1.0)*∂IV_∂λ + IV 
			end
		end

		if flags[:alpha]
			W = sum(Wk[1,:].*alpha)
			if nest_num == nest_star
				grad[idx[:alpha]] = Wk[1,:]
			end
			push!(D, W + λ_k*IV)
			push!(∂D_∂α, Wk[1,:])
		else
			push!(D, λ_k*IV)
		end
		push!(∂D_∂β, λ_k*∂IV_∂β)
		push!(∂D_∂λ, IV + λ_k*∂IV_∂λ)

	end 
	∂lnG_∂D = multinomial(D)
	∂lnG_∂β = sum((∂lnG_∂D.*∂D_∂β)[l]  for l in 1:length(D))
	∂lnG_∂λ = ∂lnG_∂D.*∂D_∂λ 
	if flags[:alpha]
		∂lnG_∂α = sum([∂lnG_∂D[l].*∂D_∂α[l]  for l in 1:length(D)]) 
		grad[idx[:alpha]] .-= ∂lnG_∂α
	end
	# Gradient
	grad[idx[:beta]] .-= ∂lnG_∂β
	if Numλ>1
		grad[idx[:lambda][nest_idx]] .-= ∂lnG_∂λ 
	else 
		grad[idx[:lambda][1]] -= sum(∂lnG_∂λ)
	end

	return -grad
end

analytic_grad_nlogit_case(x::Vector{Float64}, model::nlogit_model, data::nlogit_data, id::Int64) = 
	analytic_grad_nlogit_case(x, model.params, data, id, model.flags, model.idx, model.opts[:RUM])

function hessian_nlogit_case(theta::Vector{T}, model::nlogit_model, data::nlogit_data, id::Int64) where T<:Real
	ForwardDiff.hessian(x->ll_nlogit_case(x,model, data, id), theta)
end

function fg_nlogit_case(theta::Vector{T}, model::nlogit_model, data::nlogit_data, id::Int64) where T<:Real
	nlogit_case(ll_nlogit_case(theta, model, data, id), grad_nlogit_case(theta, model, data, id), Matrix{T}(undef,0,0)) 
end

function analytic_fg_nlogit_case(x::Vector{Float64}, θ::nlogit_param, nld::nlogit_data, id::Int64, 
										flags::Dict, idx::Dict, RUM::Bool)  
	vec_to_theta!(x, θ, flags, idx)	
	@unpack beta, alpha, lambda = θ

	Nbeta = length(beta)
	Nalpha = length(alpha)
	Numλ = length(lambda)	

	## Allocate memory
	grad = zeros(Nbeta + Nalpha + Numλ)

	D = Vector{Float64}()
	if Nbeta>1 
		∂D_∂β = VV{Float64}()
	else 
		∂D_∂β = Vector{Float64}()
	end 
	if flags[:alpha] & Nalpha>1 
		∂D_∂α =VM{Float64}()
	else 
		∂D_∂α = VV{Float64}()
	end 
	∂D_∂λ = Vector{Float64}()
	nest_idx = Int64[]
	LL = 0.
	## Compute gradient loop over nest first, then construct across nest components of ll afterwards
	for nest_data in nld[id]

		@unpack jstar, dstar, nest_star, nest_num, Xj, Wk = nest_data

		J,K = size(Xj)

		if Numλ > 1 
			λ_k = RUM ? fun_lambda(lambda[nest_num]) : lambda[nest_num]
		else 
			λ_k = RUM ? fun_lambda(lambda[1]) : lambda[1]
		end
			
		V = Xj*beta /λ_k 	# Jl by 1
		∂V_∂β = Xj ./λ_k 	# J x K of ∂Vj_∂βk = ∂V_∂β[j,k]
		∂V_∂λ = -V ./λ_k 	# Jl by 1 

		IV = logsumexp(V)  	# Scalar
		∂IV_∂V = multinomial(V)  # Jl by 1
		∂IV_∂β = sum(repeat(∂IV_∂V, 1, K).*∂V_∂β,dims=1)[:]
		∂IV_∂λ = sum(∂IV_∂V.*∂V_∂λ, dims=1)[1]

		push!(nest_idx, nest_num)

		if flags[:alpha]
			W = sum(Wk[1,:].*alpha)
			if nest_num == nest_star
				LL += V[jstar] + W + (λ_k - 1.0)*IV
				grad[idx[:alpha]] = Wk[1,:]
			end
			push!(D, W + λ_k*IV)
			push!(∂D_∂α, Wk[1,:])
		else
			if nest_num == nest_star
				LL += V[jstar] + (λ_k - 1.0)*IV
			end
			push!(D, λ_k*IV)
		end
		if nest_num == nest_star
			grad[idx[:beta]] .+= view(∂V_∂β,jstar,:) .+ (λ_k - 1.0).*∂IV_∂β 
			if Numλ > 1 
				grad[idx[:lambda][nest_num]] += ∂V_∂λ[jstar] + (λ_k - 1.0)*∂IV_∂λ + IV 
			else 
				grad[idx[:lambda][1]] += ∂V_∂λ[jstar] + (λ_k - 1.0)*∂IV_∂λ + IV 
			end
		end
		push!(∂D_∂β, λ_k*∂IV_∂β)
		push!(∂D_∂λ, IV + λ_k*∂IV_∂λ)
	end 
	LL -= logsumexp(D)
	∂lnG_∂D = multinomial(D)
	∂lnG_∂β = sum((∂lnG_∂D.*∂D_∂β)[l]  for l in 1:length(D))
	∂lnG_∂λ = ∂lnG_∂D.*∂D_∂λ 
	if flags[:alpha]
		∂lnG_∂α = sum([∂lnG_∂D[l].*∂D_∂α[l]  for l in 1:length(D)]) 
		grad[idx[:alpha]] .-= ∂lnG_∂α
	end
	# Gradient
	grad[idx[:beta]] .-= ∂lnG_∂β
	if Numλ>1
		grad[idx[:lambda][nest_idx]] .-= ∂lnG_∂λ 
	else 
		grad[idx[:lambda][1]] -= sum(∂lnG_∂λ)
	end

	return nlogit_case(-LL, -grad, Matrix{Float64}(undef, 0,0))
end

analytic_fg_nlogit_case(x::Vector{Float64}, model::nlogit_model, data::nlogit_data, id::Int64) = 
	analytic_fg_nlogit_case(x, model.params, data, id, model.flags, model.idx, model.opts[:RUM])


function fgh_nlogit_case(theta::Vector{T}, model::nlogit_model, data::nlogit_data, id::Int64) where T<:Real
	nlogit_case(ll_nlogit_case(theta, model, data, id), grad_nlogit_case(theta, model, data, id), hessian_nlogit_case(theta, model, data, id)) 
end

function estimate_nlogit(model::nlogit_model, data::nlogit_data; 
							opt_mode = :serial, opt_method = :none, grad_type = :analytic,
							x_initial = randn(model.nx), algorithm = LBFGS(), batch_size=1,
							optim_opts = Optim.Options(), workers=workers())
	
	clos_ll_nlogit_case(pd) = ll_nlogit_case(pd.theta, model, data, pd.id)
	clos_grad_nlogit_case(pd) = grad_nlogit_case(pd.theta, model, data, pd.id)
	clos_analytic_grad_nlogit_case(pd) = analytic_grad_nlogit_case(pd.theta, model, data, pd.id)
	clos_hessian_nlogit_case(pd) = hessian_nlogit_case(pd.theta, model, data, pd.id)
	clos_fg_nlogit_case(pd) = fg_nlogit_case(pd.theta, model, data, pd.id)
	clos_analytic_fg_nlogit_case(pd) = analytic_fg_nlogit_case(pd.theta, model, data, pd.id)
	clos_fgh_nlogit_case(pd) = fgh_nlogit_case(pd.theta, model, data, pd.id)
	
	if opt_mode == :parallel
		pool = CachingPool(workers)

		function pmap_nlogit_ll(theta::Vector{T}) where T<:Real 
			PD = [passdata(theta, i) for i in 1:length(data)]	
			sum(pmap(clos_ll_nlogit_case, pool, PD, batch_size=batch_size))
		end

		function pmap_nlogit_analytic_grad(theta::Vector{T}) where T<:Real
			PD = [passdata(theta, i) for i in 1:length(data)]	
			sum(pmap(clos_analytic_grad_clogit_case, pool, PD, batch_size=batch_size))
		end

		function pmap_nlogit_analytic_fg!(F, G, theta::Vector{T}) where T<:Real
			PD = [passdata(theta, i) for i in 1:length(data)]	
			nlc = pmap(clos_analytic_fg_nlogit_case, pool, PD, batch_size=batch_size)
			if G != nothing
				G[:] = sum([y.G for y in nlc])
			end
			if F != nothing
				return sum([y.F for y in nlc])
			end
		end

		function pmap_nlogit_grad(theta::Vector{T}) where T<:Real
			PD = [passdata(theta, i) for i in 1:length(data)]	
			sum(pmap(clos_grad_clogit_case, pool, PD, batch_size=batch_size))
		end

		function pmap_nlogit_Hess(theta::Vector{T}) where T<:Real
			PD = [passdata(theta, i) for i in 1:length(data)]	
			sum(pmap(clos_hessian_nlogit_case, pool, PD, batch_size=batch_size))
		end

		function pmap_nlogit_fg!(F, G, theta::Vector{T}) where T<:Real
			PD = [passdata(theta, i) for i in 1:length(data)]	
			nlc = pmap(clos_fg_nlogit_case, pool, PD, batch_size=batch_size)
			if G != nothing
				G[:] = sum([y.G for y in nlc])
			end
			if F != nothing
				return sum([y.F for y in nlc])
			end
		end

		function pmap_nlogit_fgh!(F, G, H, theta::Vector{T}) where T<:Real
			PD = [passdata(theta, i) for i in 1:length(data)]	
			nlc = pmap(clos_fgh_nlogit_case, pool, PD, batch_size=batch_size)
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
			if grad_type == :analytic
				out = Optim.optimize(Optim.only_fg!(pmap_nlogit_analytic_fg!), x_initial, algorithm, optim_opts)
			else 
				out = Optim.optimize(Optim.only_fg!(pmap_nlogit_fg!), x_initial, algorithm, optim_opts)
			end
		elseif opt_method == :hess
			out = Optim.optimize(Optim.only_fgh!(pmap_nlogit_fgh!), x_initial, algorithm, optim_opts)
		else 
			out = Optim.optimize(pmap_nlogit_ll, x_initial, NelderMead(), optim_opts)
		end

	elseif opt_mode == :serial 
		
		function map_nlogit_ll(theta::Vector{T}) where T<:Real 
			PD = [passdata(theta, i) for i in 1:length(data)]	
			sum(map(clos_ll_nlogit_case, PD))
		end

		function map_nlogit_analytic_grad(theta::Vector{T}) where T<:Real
			PD = [passdata(theta, i) for i in 1:length(data)]	
			sum(map(clos_analytic_grad_nlogit_case, PD))
		end
		
		function map_nlogit_analytic_fg!(F, G, theta::Vector{T}) where T<:Real
			PD = [passdata(theta, i) for i in 1:length(data)]	
			nlc = map(clos_analytic_fg_nlogit_case, PD)
			if G != nothing
				G[:] = sum([y.G for y in nlc])
			end
			if F != nothing
				return sum([y.F for y in nlc])
			end
		end

		function map_nlogit_grad(theta::Vector{T}) where T<:Real
			PD = [passdata(theta, i) for i in 1:length(data)]	
			sum(map(clos_grad_nlogit_case, PD))
		end

		function map_nlogit_fg!(F, G, theta::Vector{T}) where T<:Real
			PD = [passdata(theta, i) for i in 1:length(data)]	
			nlc = map(clos_fg_nlogit_case, PD)
			if G != nothing
				G[:] = sum([y.G for y in nlc])
			end
			if F != nothing
				return sum([y.F for y in nlc])
			end
		end

		function map_nlogit_Hess(theta::Vector{T}) where T<:Real
			PD = [passdata(theta, i) for i in 1:length(data)]	
			sum(map(clos_hessian_nlogit_case, PD))
		end

		function map_nlogit_fgh!(F, G, H, theta::Vector{T}) where T<:Real
			PD = [passdata(theta, i) for i in 1:length(data)]	
			nlc = map(clos_fgh_nlogit_case, PD)
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
			if grad_type == :analytic
				out = Optim.optimize(Optim.only_fg!(map_nlogit_analytic_fg!), x_initial, algorithm, optim_opts)
			else 
				out = Optim.optimize(Optim.only_fg!(map_nlogit_fg!), x_initial, algorithm, optim_opts)
			end
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

function estimate_nlogit(nl::nlogit; opt_mode = :serial, opt_method = :none, grad_type = :analytic,
						x_initial = randn(nl.model.nx), algorithm = LBFGS(), batch_size=1,
						optim_opts = Optim.Options(), workers=workers())
	estimate_nlogit(nl.model, nl.data; opt_mode = opt_mode, opt_method = opt_method, grad_type = grad_type,
					x_initial = x_initial , algorithm = algorithm, batch_size=batch_size,
					optim_opts = optim_opts, workers=workers)
end