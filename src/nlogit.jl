# Two-level nested logit

# Multiple dispatch to omit nest specific variable (i.e. logit + nest parameters)
function nlogit_model(f_beta::StatsModels.FormulaTerm, df::DataFrame; case::Symbol, nests::Symbol, choice_id::Symbol, RUM::Bool=false, num_lambda::Int64=1, nest_labels=levels(df[nests]))
	f_alpha = @eval(@formula($(f_beta.lhs.sym) ~ 0))

	f_beta = StatsModels.apply_schema(f_beta, StatsModels.schema(f_beta, df))
	f_alpha = StatsModels.apply_schema(f_alpha, StatsModels.schema(f_alpha, df))
	NX = isa(f_beta.rhs.terms[1], StatsModels.InterceptTerm{false}) ? 0 : size(modelcols(f_beta , df)[2], 2)
	NW = isa(f_alpha.rhs.terms[1], StatsModels.InterceptTerm{false}) ? 0 : size(modelcols(f_alpha , df)[2], 2) 
	np = nlogit_param(NX, NW, num_lambda )
	opts = Dict(:RUM=>RUM, :num_lambda=>num_lambda, :outside_share=>0.)
	flags = Dict(:beta=>true, :alpha=>false, :lambda=>true)
	idx, nx = get_vec_dict(np, flags)
	coefnms = num_lambda>1 ? 
		vcat(coefnames(f_beta)[2],["λ_$(i)" for i in nest_labels]) :
		vcat(coefnames(f_beta)[2],["λ"])
	return nlogit_model(f_beta, f_alpha, np, case, nests, choice_id, idx, nx, coefnms, opts, flags)
end

# Multiple dispatch to include nest specific variable (i.e. logit + nest parameters)
function nlogit_model(f_beta::StatsModels.FormulaTerm, f_alpha::StatsModels.FormulaTerm, df::DataFrame; case::Symbol, nests::Symbol, choice_id::Symbol, RUM::Bool=false, num_lambda::Int64=1, nest_labels=levels(df[nests])) 
	f_beta = StatsModels.apply_schema(f_beta, StatsModels.schema(f_beta, df))
	f_alpha = StatsModels.apply_schema(f_alpha, StatsModels.schema(f_alpha, df))
	NX = isa(f_beta.rhs.terms[1], StatsModels.InterceptTerm{false}) ? 0 : size(modelcols(f_beta , df)[2], 2)
	NW = isa(f_alpha.rhs.terms[1], StatsModels.InterceptTerm{false}) ? 0 : size(modelcols(f_alpha , df)[2], 2) 
	np = nlogit_param(NX, NW, num_lambda)
	opts = Dict(:RUM=>RUM, :num_lambda=>num_lambda, :outside_share=>0.)
	flags = Dict(:beta=>true, :alpha=>true, :lambda=>true)
	idx, nx = get_vec_dict(np, flags)
	coefnms = num_lambda>1 ? 
		vcat(coefnames(f_beta)[2],coefnames(f_alpha)[2],["λ_$(i)" for i in nest_labels]) :
		vcat(coefnames(f_beta)[2],coefnames(f_alpha)[2],["λ"])
	return nlogit_model(f_beta, f_alpha, np, case, nests, choice_id, idx, nx, coefnms, opts, flags)
end

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
			nestnum = nestdf[1,nest_id]
			if nestnum == nest_star
				jstar = findall(x->x==1, nestdf[f_beta.lhs.sym])[1]
			else
				jstar = 0
			end
			Xj = isa(f_beta.rhs.terms[1], StatsModels.InterceptTerm{false}) ?  Matrix{Float64}(undef, 0, 0) : modelcols(f_beta , nestdf)[2]
			Wk = isa(f_alpha.rhs.terms[1], StatsModels.InterceptTerm{false}) ? Matrix{Float64}(undef, 0, 0) : modelcols(f_alpha , nestdf)[2] 
			groupid = nestdf[1, case_id][1]
			push!(casedata, nlogit_nest_data(groupid, nestdf[choice_id][:], jstar, dstar, nest_star, nestnum, Xj, Wk ))
		end
		push!(dataset, casedata)
	end
	return dataset
end

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
vec_to_theta!(x::Vector{Float64}, model::nlogit_model) = vec_to_theta!(x, model.params, model.flags, model.idx)

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
			@unpack case_num, jid, jstar, dstar, nest_star, nest_num, Xj, Wk = nest_data
			if Numλ > 1 
				λ_k =  lambda[nest_num]
			else 
				λ_k =  lambda[1]
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
function ll_nlogit_case(x::Vector{T}, θ::nlogit_param, nlnd::nlogit_case_data, flags::Dict, idx::Dict, RUM::Bool) where T<:Real 
	vec_to_theta!(x, θ, flags, idx)	
	@unpack beta, alpha, lambda = θ
	Numλ = length(lambda)	
	LL = 0.
	D = Vector{Real}()
	for nest_data in nlnd
		@unpack case_num, jid, jstar, dstar, nest_star, nest_num, Xj, Wk = nest_data
		if Numλ > 1 
			λ_k = lambda[nest_num]
		else 
			λ_k = lambda[1]
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

ll_nlogit_case(x::Vector{T},  model::nlogit_model, data::nlogit_case_data) where T<:Real = 
	ll_nlogit_case(x, model.params, data, model.flags, model.idx, model.opts[:RUM])
ll_nlogit_case(x::Vector{T}, nl::nlogit, id::Int64) where T<:Real = ll_nlogit_case(x, nl.model, nl.data[id])

function grad_nlogit_case(theta::Vector{T}, model::nlogit_model, data::nlogit_case_data) where T<:Real
	ForwardDiff.gradient(x->ll_nlogit_case(x, model, data), theta)
end
grad_nlogit_case(theta::Vector{T}, nl::nlogit, id::Int64) where T<:Real = grad_nlogit_case(theta, nl.model, nl.data[id])

function analytic_grad_nlogit_case(x::Vector{Float64}, θ::nlogit_param, data::nlogit_case_data, flags::Dict, idx::Dict, RUM::Bool)  
	
	small = 1e-300
	vec_to_theta!(x, θ, flags, idx)	
	@unpack beta, alpha, lambda = θ

	Nbeta = length(beta)
	Nalpha = length(alpha)
	Numλ = length(lambda)	

	## Allocate memory
	grad = zeros(Nbeta + Nalpha + Numλ)

	D = Vector{Float64}()
	#if Nbeta>1 
	∂D_∂β = VV{Float64}()
	#else 
	#	∂D_∂β = Vector{Float64}()
	#end 
	if flags[:alpha] & Nalpha>1 
		∂D_∂α =VM{Float64}()
	else 
		∂D_∂α = VV{Float64}()
	end 
	∂D_∂λ = Vector{Float64}()
	nest_idx = Int64[]

	## Compute gradient loop over nest first, then construct across nest components of ll afterwards
	for nest_data in data

		@unpack case_num, jid, jstar, dstar, nest_star, nest_num, Xj, Wk = nest_data

		J,K = size(Xj)

		if Numλ > 1 
			λ_k = lambda[nest_num]
		else 
			λ_k = lambda[1]
		end
			
		V = Xj*beta /λ_k 	# Jl by 1
		∂V_∂β = Xj ./λ_k 	# J x K of ∂Vj_∂βk = ∂V_∂β[j,k]
		∂V_∂λ = -V ./λ_k 	# Jl by 1 

		IV = logsumexp(V)  	# Scalar
		∂IV_∂V = max.(small, multinomial(V))  # Jl by 1
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
	∂lnG_∂D = max.(small, multinomial(D))
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

analytic_grad_nlogit_case(x::Vector{Float64}, model::nlogit_model, data::nlogit_case_data) = 
	analytic_grad_nlogit_case(x, model.params, data, model.flags, model.idx, model.opts[:RUM])
analytic_grad_nlogit_case(x::Vector{Float64}, nl::nlogit, id::Int64) = analytic_grad_nlogit_case(x, nl.model, nl.data[id])

function fg_nlogit_case(theta::Vector{T}, model::nlogit_model, data::nlogit_case_data) where T<:Real
	nlogit_case(ll_nlogit_case(theta, model, data), grad_nlogit_case(theta, model, data), Matrix{T}(undef,0,0)) 
end

function analytic_fg_nlogit_case(x::Vector{Float64}, θ::nlogit_param, data::nlogit_case_data,
										flags::Dict, idx::Dict, RUM::Bool)  
	small = 1e-300
	vec_to_theta!(x, θ, flags, idx)	
	@unpack beta, alpha, lambda = θ

	Nbeta = length(beta)
	Nalpha = length(alpha)
	Numλ = length(lambda)	

	## Allocate memory
	grad = zeros(Nbeta + Nalpha + Numλ)

	D = Vector{Float64}()
	#if Nbeta>1 
	∂D_∂β = VV{Float64}()
	#else 
	#	∂D_∂β = Vector{Float64}()
	#end 
	if flags[:alpha] & Nalpha>1 
		∂D_∂α =VM{Float64}()
	else 
		∂D_∂α = VV{Float64}()
	end 
	∂D_∂λ = Vector{Float64}()
	nest_idx = Int64[]
	LL = 0.
	## Compute gradient loop over nest first, then construct across nest components of ll afterwards
	for nest_data in data

		@unpack case_num, jid, jstar, dstar, nest_star, nest_num, Xj, Wk = nest_data

		J,K = size(Xj)

		if Numλ > 1 
			λ_k = lambda[nest_num]
		else 
			λ_k = lambda[1]
		end
			
		V = Xj*beta /λ_k 	# Jl by 1
		∂V_∂β = Xj ./λ_k 	# J x K of ∂Vj_∂βk = ∂V_∂β[j,k]
		∂V_∂λ = -V ./λ_k 	# Jl by 1 

		IV = logsumexp(V)  	# Scalar
		∂IV_∂V = max.(small, multinomial(V) ) # Jl by 1
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
	∂lnG_∂D = max.(small, multinomial(D))
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

analytic_fg_nlogit_case(x::Vector{Float64}, model::nlogit_model, data::nlogit_case_data) = 
	analytic_fg_nlogit_case(x, model.params, data, model.flags, model.idx, model.opts[:RUM])

analytic_fg_nlogit_case(x::Vector{Float64}, nl::nlogit, id::Int64) = analytic_fg_nlogit_case(x, nl.model, nl.data[id])

function hessian_nlogit_case(theta::Vector{T}, model::nlogit_model, data::nlogit_case_data) where T<:Real
	ForwardDiff.hessian(x->ll_nlogit_case(x,model, data), theta)
end
hessian_nlogit_case(theta::Vector{T}, nl::nlogit, id::Int64) where T<:Real = hessian_nlogit_case(theta, nl.model, nl.data[id])

function pmap_hessian_nlogit(theta::Vector{T}, data::nlogit_data; batch_size = 1) where T<:Real
	DD = [distdata(theta, nld) for nld in data]	
	sum(pmap(hessian_nlogit_case, DD, batch_size=batch_size))
end
pmap_hessian_nlogit(theta::Vector{T}, nl::nlogit; batch_size = 1) where T<:Real = pmap_hessian_nlogit(theta, nl.data; batch_size = batch_size)

function fgh_nlogit_case(theta::Vector{T}, model::nlogit_model, data::nlogit_case_data) where T<:Real
	nlogit_case(ll_nlogit_case(theta, model, data), grad_nlogit_case(theta, model, data), hessian_nlogit_case(theta, model, data)) 
end

fgh_nlogit_case(theta::Vector{T}, nl::nlogit, id::Int64) where T<:Real = fgh_nlogit_case(theta, nl.model, nl.data[id])


function nlogit_prob(x::Vector{T}, θ::nlogit_param, nlnd::nlogit_case_data, flags::Dict, idx::Dict, 
		RUM::Bool, outside_share::Float64, case_id::Symbol, nest_id::Symbol, choice_id::Symbol) where T<:Real
	small = 1e-300
	vec_to_theta!(x, θ, flags, idx)	
	@unpack beta, alpha, lambda = θ

	Nbeta = length(beta)
	Nalpha = length(alpha)
	Numλ = length(lambda)	

	## Allocate memory
	grad = zeros(Nbeta + Nalpha + Numλ)

	denom = Vector{Real}()
	s_jg = VV{eltype(x)}()
	s_j = VV{eltype(x)}()
	for nest_data in nlnd
		@unpack case_num, jid, jstar, dstar, nest_star, nest_num, Xj, Wk = nest_data
		if Numλ > 1 
			λ_k = lambda[nest_num]
		else 
			λ_k = lambda[1]
		end
		V = Xj*beta /λ_k
		push!(s_jg , max.(small, multinomial(V)) )
		if flags[:alpha]
			W = Wk*alpha
			push!(denom, W[1] + λ_k*logsumexp(V))	
		else 
			push!(denom, λ_k*logsumexp(V))	
		end		
	end
	s_g = max.(small,multinomial(denom))
	DF = DataFrame[]
	for (g, nest_data) in enumerate(nlnd)
		@unpack case_num, jid, jstar, dstar, nest_star, nest_num, Xj, Wk = nest_data
		J = length(jid)
		case = case_num*ones(Int64,J)
		nst = nest_num*ones(Int64,J)
		sj = (1.0 .- outside_share).*s_jg[g].*s_g[g]
		sg = (1.0 .- outside_share).*s_g[g].*ones(J)
		push!(DF, DataFrame(case_id => case, nest_id => nst, choice_id => jid, :pr_j => sj, :pr_jg => s_jg[g], :pr_g => sg))
 	end

 	outdf = deepcopy(DF[1])
	@inbounds for n in 2:length(DF)
		append!(outdf, DF[n])
	end

	return outdf
end

nlogit_prob(x::Vector{T}, model::nlogit_model, data::nlogit_case_data) where T<:Real = 
	nlogit_prob(x, model.params, data, model.flags, model.idx, model.opts[:RUM], model.opts[:outside_share], model.case_id, model.nest_id, model.choice_id)

nlogit_prob(x::Vector{T}, nl::nlogit, case_num::Int64) where T<:Real = nlogit_prob(x, nl.model, nl.data[case_num])

function nlogit_prob(x::Vector{T}, nl::nlogit) where T<:Real
	DF = DataFrame[]
	for case_data in nl.data
		push!(DF, nlogit_prob(x, nl.model, case_data))
	end
	outdf = deepcopy(DF[1])
	@inbounds for n in 2:length(DF)
		append!(outdf, DF[n])
	end
	return outdf
end	

