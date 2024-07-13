
function elas_own_nlogit(x::Vector{T}, θ::nlogit_param, nlnd::nlogit_case_data, 
							flags::Dict, idx::Dict, RUM::Bool, outside_share::Float64, 
								case_id::Symbol, nest_id::Symbol, choice_id::Symbol, xvar_indices::Vector{Int64}) where T<:Real
	
	vec_to_theta!(x, θ, flags, idx)	
	@unpack beta, alpha, lambda = θ

	Numλ = length(lambda)	

	df = nlogit_prob(x, θ, nlnd, flags, idx, RUM, outside_share, case_id, nest_id, choice_id)

	EDF = DataFrame[]
	for nest_data in nlnd 
		@unpack case_num, jid, jstar, dstar, nest_star, nest_num, Xj, Wk = nest_data
		J = size(Xj,1)

		# Step 0: Load identifiers & match with choice prob
		elasdf = DataFrame(case_id=>case_num*ones(J), nest_id=>nest_num*ones(J), choice_id=>jid)
		elasdf = leftjoin(elasdf, df, on=[case_id, nest_id, choice_id])

		# Step 1: Price Terms
		alpha_x_xvar = zeros(eltype(x),J)
		for inds in xvar_indices
			alpha_x_xvar .+= Xj[:,inds].*beta[inds]
		end			
			
		# Step 2: Get nest parameter
		λg = Numλ > 1 ? lambda[nest_num] : lambda[1]

		# Step 3: Calculate elasticity
		elasdf[!, :ejj] = alpha_x_xvar.*( 1.0 .- λg.*elasdf.pr_j .- (1.0 .- λg).*elasdf.pr_jg) ./λg

		# Step 4: Move Dataframe to storage vector 
		push!(EDF, elasdf)

	end

	# Compile results into a Dataframe for the case
	outdf = deepcopy(EDF[1])
	@inbounds for n in 2:length(EDF)
		append!(outdf, EDF[n])
	end

	return outdf

end

elas_own_nlogit(x::Vector{T}, model::nlogit_model, nlnd::nlogit_case_data, xvar_indices::Vector{Int64}) where T<:Real =
	elas_own_nlogit(x, model.params, nlnd, model.flags, model.idx, model.opts[:RUM], model.opts[:outside_share], model.case_id, model.nest_id, model.choice_id, xvar_indices) 

function elas_own_nlogit(x::Vector{T}, model::nlogit_model, nld::nlogit_data, xvar_indices::Vector{Int64}) where T<:Real
	EDF = DataFrame[]
	for case_data in nld
		push!(EDF, elas_own_nlogit(x, model, case_data, xvar_indices))
	end
	outdf = deepcopy(EDF[1])
	@inbounds for n in 2:length(EDF)
		append!(outdf, EDF[n])
	end
	return outdf
end	

elas_own_nlogit(x::Vector{T}, nl::nlogit, xvar_indices::Vector{Int64}) where T<:Real = elas_own_nlogit(x, nl.model, nl.data, xvar_indices)

# ------------- Gradient of elasticities --------------- #

function grad_elas_own_nlogit(x::Vector{T}, model::nlogit_model, nlnd::nlogit_case_data, xvar_indices::Vector{Int64}) where T<:Real
	
	# Ease of reference
	idx = deepcopy(model.idx)
	nest_id = model.nest_id
	outside_share = model.opts[:outside_share]

	# Code below here
	vec_to_theta!(x, model.params, model.flags, idx)	
	@unpack beta, alpha, lambda = model.params

	Nbeta = length(beta)
	Nalpha = length(alpha)
	Numλ = length(lambda)	
	Nx = Nbeta + Nalpha + Numλ

	df_probs = nlogit_prob(x, model, nlnd);
	(∇s_j,∇s_jg, ∇s_g)= grad_nlogit_prob(x, model, nlnd);	

	λg = lambda[df_probs[nest_id]]
	s_j = df_probs[:pr_j]
	s_jg = df_probs[:pr_jg]
	s_g = df_probs[:pr_g]

	X = Real[]
	for (ctr, nest_data) in enumerate(nlnd) 
		append!(X , nest_data.Xj)
	end
	J = length(λg);
	K = size(nlnd[1].Xj, 2);
	X = reshape(X, J, K)

	alpha_x_xvar = X[:,xvar_indices]*x[xvar_indices]

	LT = X[:,xvar_indices]*x[xvar_indices]./λg
	∂LT_∂β = [zeros(Nbeta) for j in 1:J]
	∂LT_∂α = Nalpha>0 ? [zeros(Nalpha) for j in 1:J] : []
	∂LT_∂λ = [zeros(Numλ) for j in 1:J]

	RT = 1.0 .- λg.*s_j .- (1.0 .- λg).*s_jg
	∂RT_∂β = [zeros(Nbeta) for j in 1:J]
	∂RT_∂α = Nalpha>0 ? [zeros(Nalpha) for j in 1:J] : []
	∂RT_∂λ = [zeros(Numλ) for j in 1:J]

	if Nalpha==0
		for j in 1:J
			# λg[j] = lambda[g]
			g =  df_probs[j,nest_id]			
			∂LT_∂β[j] .+= vec(X[idx[:beta],xvar_indices]./λg[j])
			∂RT_∂β[j] .+= -λg[j].*∇s_j[j][idx[:beta]] .- (1.0 .- λg[j]).*∇s_jg[j][idx[:beta]]
			for l in 1:Numλ
				if l==g
					∂LT_∂λ[j][g] = -LT[idx[:lambda][g]]./λg[j]
					∂RT_∂λ[j][g] += -s_j[j] .- λg[j].*∇s_j[j][idx[:lambda][g]] .- (1.0 .- λg[j]).*∇s_jg[j][idx[:lambda][g]] .+ s_jg[j]
				else 
					∂RT_∂λ[j][l] += - λg[j].*∇s_j[j][idx[:lambda][l]] .- (1.0 .- λg[j]).*∇s_jg[j][idx[:lambda][l]]
				end
			end
		end 
		∂ejj_∂β = ∂LT_∂β .* RT .+ LT .* ∂RT_∂β 
		∂ejj_∂λ = ∂LT_∂λ .* RT .+ LT .* ∂RT_∂λ
		return [[∂ejj_∂β[j]; ∂ejj_∂λ[j] ] for j in 1:J]
	else
		for j in 1:J
			g =  df_probs[j,nest_id]			
			∂LT_∂β[j] .+= vec(X[idx[:beta],xvar_indices]./λg[j])
			∂RT_∂β[j] .+= -λg[j].*∇s_j[j][idx[:beta]] .- (1.0 .- λg[j]).*∇s_jg[j][idx[:beta]]
			∂RT_∂α[j] .+= -λg[j].*∇s_j[j][idx[:alpha]] .- (1.0 .- λg[j]).*∇s_jg[j][idx[:alpha]]
			for l in 1:Numλ
				if l==g
					∂LT_∂λ[j][g] = -LT[idx[:lambda][g]]./λg[j]
					∂RT_∂λ[j][g] += -s_j[j] .- λg[j].*∇s_j[j][idx[:lambda][g]] .- (1.0 .- λg[j]).*∇s_jg[j][idx[:lambda][g]] .+ s_jg[j]
				else 
					∂RT_∂λ[j][l] += - λg[j].*∇s_j[j][idx[:lambda][l]] .- (1.0 .- λg[j]).*∇s_jg[j][idx[:lambda][l]]
				end
			end
		end 
		∂ejj_∂β = ∂LT_∂β * RT + LT * ∂RT_∂β 
		∂ejj_∂α = LT * ∂RT_∂α 
		∂ejj_∂λ = ∂LT_∂λ * RT + LT * ∂RT_∂λ
		return [[∂ejj_∂β[j]; ∂ejj_∂α[j]; ∂ejj_∂λ[j] ] for j in 1:J]
	end

end

function grad_elas_own_nlogit(x::Vector{T}, model::nlogit_model, nld::nlogit_data, xvar_indices::Vector{Int64}) where T<:Real
	EJJ = Float64[]
	for case_data in nld
		append!(EJJ, grad_elas_own_nlogit(x, model, case_data, xvar_indices))
	end
	return EJJ
end


# within nest cross elasticities


function elas_within_nlogit(x::Vector{T}, θ::nlogit_param, nlnd::nlogit_case_data, 
							flags::Dict, idx::Dict, RUM::Bool, outside_share::Float64, 
								case_id::Symbol, nest_id::Symbol, choice_id::Symbol, xvar_indices::Vector{Int64}) where T<:Real
	vec_to_theta!(x, θ, flags, idx)	
	@unpack beta, alpha, lambda = θ

	Numλ = length(lambda)	

	df = nlogit_prob(x, θ, nlnd, flags, idx, RUM, outside_share, case_id, nest_id, choice_id)

	EDF = DataFrame[]
	for nest_data in nlnd 
		@unpack case_num, jid, jstar, dstar, nest_star, nest_num, Xj, Wk = nest_data
		J = size(Xj,1)
		
		# Step 0: Load identifiers & match with choice prob
		elasdf = DataFrame(case_id=>case_num*ones(J), nest_id=>nest_num*ones(J), choice_id=>jid)
		elasdf = leftjoin(elasdf, df, on=[case_id, nest_id, choice_id])

		# Step 1: Price Terms
		alpha_x_xvar = zeros(eltype(x),J)
		for inds in xvar_indices
			alpha_x_xvar .+= Xj[:,inds].*beta[inds]
		end			
			
		# Step 2: Get nest parameter
		λg = Numλ > 1 ? lambda[nest_num] : lambda[1]

		# Step 3: Calculate elasticity
		elasdf[!, :ekjg] = -alpha_x_xvar.*( λg.*elasdf.pr_j .+ (1.0 .- λg).*elasdf.pr_jg ) ./λg

		# Step 4: Move Dataframe to storage vector 
		push!(EDF, elasdf)

	end

	# Compile results into a Dataframe for the case
	outdf = deepcopy(EDF[1])
	@inbounds for n in 2:length(EDF)
		append!(outdf, EDF[n])
	end

	return outdf

end

elas_within_nlogit(x::Vector{T}, model::nlogit_model, nlnd::nlogit_case_data, xvar_indices::Vector{Int64}) where T<:Real =
	elas_within_nlogit(x, model.params, nlnd, model.flags, model.idx, model.opts[:RUM], model.opts[:outside_share], model.case_id, model.nest_id, model.choice_id, xvar_indices) 

function elas_within_nlogit(x::Vector{T}, model::nlogit_model, nld::nlogit_data, xvar_indices::Vector{Int64}) where T<:Real
	EDF = DataFrame[]
	for case_data in nld
		push!(EDF, elas_within_nlogit(x, model, case_data, xvar_indices))
	end
	outdf = deepcopy(EDF[1])
	@inbounds for n in 2:length(EDF)
		append!(outdf, EDF[n])
	end
	return outdf
end	

elas_within_nlogit(x::Vector{T}, nl::nlogit, xvar_indices::Vector{Int64}) where T<:Real = elas_within_nlogit(x, nl.model, nl.data, xvar_indices)


# ------------- Gradient of elasticities --------------- #

function grad_elas_within_nlogit(x::Vector{T}, model::nlogit_model, nlnd::nlogit_case_data, xvar_indices::Vector{Int64}) where T<:Real
	
	# Ease of reference
	idx = deepcopy(model.idx)
	nest_id = model.nest_id
	outside_share = model.opts[:outside_share]

	# Code below here
	vec_to_theta!(x, model.params, model.flags, idx)	
	@unpack beta, alpha, lambda = model.params

	Nbeta = length(beta)
	Nalpha = length(alpha)
	Numλ = length(lambda)	
	Nx = Nbeta + Nalpha + Numλ

	df_probs = nlogit_prob(x, model, nlnd);
	(∇s_j,∇s_jg, ∇s_g)= grad_nlogit_prob(x, model, nlnd);	

	λg = lambda[df_probs[nest_id]]
	s_j = df_probs[:pr_j]
	s_jg = df_probs[:pr_jg]
	s_g = df_probs[:pr_g]

	X = Real[]
	for (ctr, nest_data) in enumerate(nlnd) 
		append!(X , nest_data.Xj)
	end
	J = length(λg);
	K = size(nlnd[1].Xj, 2);
	X = reshape(X, J, K)

	alpha_x_xvar = X[:,xvar_indices]*x[xvar_indices]

	LT = -alpha_x_xvar./λg
	∂LT_∂β = [zeros(Nbeta) for j in 1:J]
	∂LT_∂α = Nalpha>0 ? [zeros(Nalpha) for j in 1:J] : []
	∂LT_∂λ = [zeros(Numλ) for j in 1:J]

	RT = λg.*s_j .+ (1.0 .- λg).*s_jg
	∂RT_∂β = [zeros(Nbeta) for j in 1:J]
	∂RT_∂α = Nalpha>0 ? [zeros(Nalpha) for j in 1:J] : []
	∂RT_∂λ = [zeros(Numλ) for j in 1:J]

	if Nalpha==0
		for j in 1:J
			# λg[j] = lambda[g]
			g =  df_probs[j,nest_id]			
			∂LT_∂β[j] .+= vec(-X[idx[:beta],xvar_indices]./λg[j])
			∂RT_∂β[j] .+= λg[j].*∇s_j[j][idx[:beta]] .+ (1.0 .- λg[j]).*∇s_jg[j][idx[:beta]]
			for l in 1:Numλ
				if l==g
					∂LT_∂λ[j][g] = -LT[idx[:lambda][g]]./λg[j]
					∂RT_∂λ[j][g] += s_j[j] .+ λg[j].*∇s_j[j][idx[:lambda][g]] .+ (1.0 .- λg[j]).*∇s_jg[j][idx[:lambda][g]] .- s_jg[j]
				else 
					∂RT_∂λ[j][l] += λg[j].*∇s_j[j][idx[:lambda][l]] .+ (1.0 .- λg[j]).*∇s_jg[j][idx[:lambda][l]]
				end
			end
		end 
		∂ekjg_∂β = ∂LT_∂β .* RT .+ LT .* ∂RT_∂β 
		∂ekjg_∂λ = ∂LT_∂λ .* RT .+ LT .* ∂RT_∂λ
		return [[∂ekjg_∂β[j]; ∂ekjg_∂λ[j] ] for j in 1:J]
	else
		for j in 1:J
			g =  df_probs[j,nest_id]			
			∂LT_∂β[j] .+= vec(-X[idx[:beta],xvar_indices]./λg[j])
			∂RT_∂β[j] .+= λg[j].*∇s_j[j][idx[:beta]] .+ (1.0 .- λg[j]).*∇s_jg[j][idx[:beta]]
			∂RT_∂α[j] .+= λg[j].*∇s_j[j][idx[:alpha]] .+ (1.0 .- λg[j]).*∇s_jg[j][idx[:alpha]]
			for l in 1:Numλ
				if l==g
					∂LT_∂λ[j][g] = -LT[idx[:lambda][g]]./λg[j]
					∂RT_∂λ[j][g] += s_j[j] .+ λg[j].*∇s_j[j][idx[:lambda][g]] .+ (1.0 .- λg[j]).*∇s_jg[j][idx[:lambda][g]] .- s_jg[j]
				else 
					∂RT_∂λ[j][l] += λg[j].*∇s_j[j][idx[:lambda][l]] .+ (1.0 .- λg[j]).*∇s_jg[j][idx[:lambda][l]]
				end
			end
		end 
		∂ekjg_∂β = ∂LT_∂β * RT + LT * ∂RT_∂β 
		∂ekjg_∂α = LT * ∂RT_∂α 
		∂ekjg_∂λ = ∂LT_∂λ * RT + LT * ∂RT_∂λ
		return [[∂ekjg_∂β[j]; ∂ekjg_∂α[j]; ∂ekjg_∂λ[j] ] for j in 1:J]
	end

end

function grad_elas_within_nlogit(x::Vector{T}, model::nlogit_model, nld::nlogit_data, xvar_indices::Vector{Int64}) where T<:Real
	EKJG = Float64[]
	for case_data in nld
		append!(EKJG, grad_elas_within_nlogit(x, model, case_data, xvar_indices))
	end
	return EKJG
end

# across nest elasticities

function elas_across_nlogit(x::Vector{T}, θ::nlogit_param, nlnd::nlogit_case_data, 
							flags::Dict, idx::Dict, RUM::Bool, outside_share::Float64, 
								case_id::Symbol, nest_id::Symbol, choice_id::Symbol, xvar_indices::Vector{Int64}) where T<:Real
	
	vec_to_theta!(x, θ, flags, idx)	
	@unpack beta, alpha, lambda = θ

	Numλ = length(lambda)	

	df = nlogit_prob(x, θ, nlnd, flags, idx, RUM, outside_share, case_id, nest_id, choice_id)

	EDF = DataFrame[]
	for nest_data in nlnd 
		@unpack case_num, jid, jstar, dstar, nest_star, nest_num, Xj, Wk = nest_data
		J = size(Xj,1)

		# Step 0: Load identifiers & match with choice prob
		elasdf = DataFrame(case_id=>case_num*ones(J), nest_id=>nest_num*ones(J), choice_id=>jid)
		elasdf = leftjoin(elasdf, df, on=[case_id, nest_id, choice_id])

		# Step 1: Price Terms
		alpha_x_xvar = zeros(eltype(x),J)
		for inds in xvar_indices
			alpha_x_xvar .+= Xj[:,inds].*beta[inds]
		end			
			
		# Step 2: Get nest parameter
		λg = Numλ > 1 ? lambda[nest_num] : lambda[1]

		# Step 3: Calculate elasticity
		elasdf[!, :ekj] = -alpha_x_xvar.*elasdf.pr_j

		# Step 4: Move Dataframe to storage vector 
		push!(EDF, elasdf)

	end

	# Compile results into a Dataframe for the case
	outdf = deepcopy(EDF[1])
	@inbounds for n in 2:length(EDF)
		append!(outdf, EDF[n])
	end

	return outdf

end


elas_across_nlogit(x::Vector{T}, model::nlogit_model, nlnd::nlogit_case_data, xvar_indices::Vector{Int64}) where T<:Real =
	elas_across_nlogit(x, model.params, nlnd, model.flags, model.idx, model.opts[:RUM], model.opts[:outside_share], model.case_id, model.nest_id, model.choice_id, xvar_indices) 

function elas_across_nlogit(x::Vector{T}, model::nlogit_model, nld::nlogit_data, xvar_indices::Vector{Int64}) where T<:Real
	EDF = DataFrame[]
	for case_data in nld
		push!(EDF, elas_across_nlogit(x, model, case_data, xvar_indices))
	end
	outdf = deepcopy(EDF[1])
	@inbounds for n in 2:length(EDF)
		append!(outdf, EDF[n])
	end
	return outdf
end	

elas_across_nlogit(x::Vector{T}, nl::nlogit, xvar_indices::Vector{Int64}) where T<:Real = elas_across_nlogit(x, nl.model, nl.data, xvar_indices)



# ------------- Gradient of elasticities --------------- #

function grad_elas_across_nlogit(x::Vector{T}, model::nlogit_model, nlnd::nlogit_case_data, xvar_indices::Vector{Int64}) where T<:Real
	
	# Ease of reference
	idx = deepcopy(model.idx)
	nest_id = model.nest_id
	outside_share = model.opts[:outside_share]

	# Code below here
	vec_to_theta!(x, model.params, model.flags, idx)	
	@unpack beta, alpha, lambda = model.params

	Nbeta = length(beta)
	Nalpha = length(alpha)
	Numλ = length(lambda)	
	Nx = Nbeta + Nalpha + Numλ

	df_probs = nlogit_prob(x, model, nlnd);
	(∇s_j,∇s_jg, ∇s_g)= grad_nlogit_prob(x, model, nlnd);	

	λg = lambda[df_probs[nest_id]]
	s_j = df_probs[:pr_j]

	X = Real[]
	for (ctr, nest_data) in enumerate(nlnd) 
		append!(X , nest_data.Xj)
	end
	J = length(λg);
	K = size(nlnd[1].Xj, 2);
	X = reshape(X, J, K)

	alpha_x_xvar = X[:,xvar_indices]*x[xvar_indices]

	LT = -alpha_x_xvar
	∂LT_∂β = [zeros(Nbeta) for j in 1:J]
	∂LT_∂α = Nalpha>0 ? [zeros(Nalpha) for j in 1:J] : []
	∂LT_∂λ = [zeros(Numλ) for j in 1:J]

	RT = s_j
	∂RT_∂β = [zeros(Nbeta) for j in 1:J]
	∂RT_∂α = Nalpha>0 ? [zeros(Nalpha) for j in 1:J] : []
	∂RT_∂λ = [zeros(Numλ) for j in 1:J]

	for j in 1:J
		∂LT_∂β[j] .+= vec(-X[idx[:beta],xvar_indices])
		∂RT_∂β[j] .+= ∇s_j[j][idx[:beta]] 
	end 
	if Nalpha==0
		∂ekj_∂β = ∂LT_∂β .* RT .+ LT .* ∂RT_∂β 
		return [[∂ekj_∂β[j]; ∂ekj_∂λ[j] ] for j in 1:J]
	else
		∂ekj_∂β = ∂LT_∂β * RT + LT * ∂RT_∂β 
		∂ekj_∂α = LT * ∂RT_∂α 
		∂ekj_∂λ = ∂LT_∂λ * RT + LT * ∂RT_∂λ
		return [[∂ekj_∂β[j]; ∂ekj_∂α[j]; ∂ekj_∂λ[j] ] for j in 1:J]
	end
end

function grad_elas_across_nlogit(x::Vector{T}, model::nlogit_model, nld::nlogit_data, xvar_indices::Vector{Int64}) where T<:Real
	EKJ = Float64[]
	for case_data in nld
		append!(EKJ, grad_elas_across_nlogit(x, model, case_data, xvar_indices))
	end
	return EKJ
end
