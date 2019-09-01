
function elas_own_nlogit(x::Vector{T}, θ::nlogit_param, nlnd::nlogit_case_data, 
							flags::Dict, idx::Dict, RUM::Bool, outside_share::Float64, 
								case_id::Symbol, nest_id::Symbol, choice_id::Symbol, price_indices::Vector{Int64}) where T<:Real
	
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
		elasdf = join(elasdf, df, on=[case_id, nest_id, choice_id], kind=:left)

		# Step 1: Price Terms
		alpha_x_price = zeros(eltype(x),J)
		for inds in price_indices
			alpha_x_price .+= Xj[:,inds].*beta[inds]
		end			
			
		# Step 2: Get nest parameter
		λg = Numλ > 1 ? lambda[nest_num] : lambda[1]

		# Step 3: Calculate elasticity
		elasdf[:ejj] = alpha_x_price.*( 1.0 .- λg.*elasdf[:pr_j] .- (1.0 .- λg).*elasdf[:pr_jg]) ./λg

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

elas_own_nlogit(x::Vector{T}, model::nlogit_model, nlnd::nlogit_case_data, price_indices::Vector{Int64}) where T<:Real =
	elas_own_nlogit(x, model.params, nlnd, model.flags, model.idx, model.opts[:RUM], model.opts[:outside_share], model.case_id, model.nest_id, model.choice_id, price_indices) 

function elas_own_nlogit(x::Vector{T}, model::nlogit_model, nld::nlogit_data, price_indices::Vector{Int64}) where T<:Real
	EDF = DataFrame[]
	for case_data in nld
		push!(EDF, elas_own_nlogit(x, model, case_data, price_indices))
	end
	outdf = deepcopy(EDF[1])
	@inbounds for n in 2:length(EDF)
		append!(outdf, EDF[n])
	end
	return outdf
end	
#=
function elas_own_nlogit(x::Vector{T}, model::nlogit_model, nld::nlogit_data, price_indices::Vector{Int64}) where T<:Real
	e_jj = eltype(x)[]
	for case_data in nld
		append!(e_jj, elas_own_nlogit(x, model, case_data, price_indices) )
	end
	return e_jj
end	
=#
elas_own_nlogit(x::Vector{T}, nl::nlogit, price_indices::Vector{Int64}) where T<:Real = elas_own_nlogit(x, nl.model, nl.data, price_indices)

# Gradient
function grad_elas_own_nlogit(x::Vector{T}, model::nlogit_model, nlnd::nlogit_case_data, price_indices::Vector{Int64}) where T<:Real
	J = 0
	for nest in nlnd 
		J += size(nest.Xj, 1)
	end
	return [ForwardDiff.gradient(θ ->elas_own_nlogit(θ,  model, nlnd, price_indices)[j], x) for j in 1:J]
end

function grad_elas_own_nlogit(x::Vector{T}, model::nlogit_model, nld::nlogit_data, price_indices::Vector{Int64}) where T<:Real
	∇e_jj = VV{eltype(x)}()
	for case_data in nld 
		append!(∇e_jj, grad_elas_own_nlogit(x, model, case_data, price_indices) )
	end
	∇e_jj
end

grad_elas_own_nlogit(x::Vector{T}, nl::nlogit, price_indices::Vector{Int64}) where T<:Real = grad_elas_own_nlogit(x, nl.model, nl.data, price_indices)


# within nest cross elasticities


function elas_within_nlogit(x::Vector{T}, θ::nlogit_param, nlnd::nlogit_case_data, 
							flags::Dict, idx::Dict, RUM::Bool, outside_share::Float64, 
								case_id::Symbol, nest_id::Symbol, choice_id::Symbol, price_indices::Vector{Int64}) where T<:Real
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
		elasdf = join(elasdf, df, on=[case_id, nest_id, choice_id], kind=:left)

		# Step 1: Price Terms
		alpha_x_price = zeros(eltype(x),J)
		for inds in price_indices
			alpha_x_price .+= Xj[:,inds].*beta[inds]
		end			
			
		# Step 2: Get nest parameter
		λg = Numλ > 1 ? lambda[nest_num] : lambda[1]

		# Step 3: Calculate elasticity
		elasdf[:ekjg] = -alpha_x_price.*( λg.*elasdf[:pr_j] .+ (1.0 .- λg).*elasdf[:pr_jg] ) ./λg

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

elas_within_nlogit(x::Vector{T}, model::nlogit_model, nlnd::nlogit_case_data, price_indices::Vector{Int64}) where T<:Real =
	elas_within_nlogit(x, model.params, nlnd, model.flags, model.idx, model.opts[:RUM], model.opts[:outside_share], model.case_id, model.nest_id, model.choice_id, price_indices) 

function elas_within_nlogit(x::Vector{T}, model::nlogit_model, nld::nlogit_data, price_indices::Vector{Int64}) where T<:Real
	EDF = DataFrame[]
	for case_data in nld
		push!(EDF, elas_within_nlogit(x, model, case_data, price_indices))
	end
	outdf = deepcopy(EDF[1])
	@inbounds for n in 2:length(EDF)
		append!(outdf, EDF[n])
	end
	return outdf
end	

elas_within_nlogit(x::Vector{T}, nl::nlogit, price_indices::Vector{Int64}) where T<:Real = elas_within_nlogit(x, nl.model, nl.data, price_indices)

# Gradient
function grad_elas_within_nlogit(x::Vector{T}, model::nlogit_model, nlnd::nlogit_case_data, price_indices::Vector{Int64}) where T<:Real
	J = 0
	for nest in nlnd 
		J += size(nest.Xj, 1)
	end
	return [ForwardDiff.gradient(θ ->elas_within_nlogit(θ,  model, nlnd, price_indices)[j], x) for j in 1:J]
end

function grad_elas_within_nlogit(x::Vector{T}, model::nlogit_model, nld::nlogit_data, price_indices::Vector{Int64}) where T<:Real
	∇e_kj = VV{eltype(x)}()
	for case_data in nld 
		append!(∇e_kj, grad_elas_within_nlogit(x, model, case_data, price_indices) )
	end
	∇e_kj
end

grad_elas_within_nlogit(x::Vector{T}, nl::nlogit, price_indices::Vector{Int64}) where T<:Real = grad_elas_within_nlogit(x, nl.model, nl.data, price_indices)


# across nest elasticities

function elas_across_nlogit(x::Vector{T}, θ::nlogit_param, nlnd::nlogit_case_data, 
							flags::Dict, idx::Dict, RUM::Bool, outside_share::Float64, 
								case_id::Symbol, nest_id::Symbol, choice_id::Symbol, price_indices::Vector{Int64}) where T<:Real
	
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
		elasdf = join(elasdf, df, on=[case_id, nest_id, choice_id], kind=:left)

		# Step 1: Price Terms
		alpha_x_price = zeros(eltype(x),J)
		for inds in price_indices
			alpha_x_price .+= Xj[:,inds].*beta[inds]
		end			
			
		# Step 2: Get nest parameter
		λg = Numλ > 1 ? lambda[nest_num] : lambda[1]

		# Step 3: Calculate elasticity
		elasdf[:ekj] = -alpha_x_price.*elasdf[:pr_j]

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

elas_across_nlogit(x::Vector{T}, model::nlogit_model, nlnd::nlogit_case_data, price_indices::Vector{Int64}) where T<:Real =
	elas_across_nlogit(x, model.params, nlnd, model.flags, model.idx, model.opts[:RUM], model.opts[:outside_share], model.case_id, model.nest_id, model.choice_id, price_indices) 

function elas_across_nlogit(x::Vector{T}, model::nlogit_model, nld::nlogit_data, price_indices::Vector{Int64}) where T<:Real
	EDF = DataFrame[]
	for case_data in nld
		push!(EDF, elas_across_nlogit(x, model, case_data, price_indices))
	end
	outdf = deepcopy(EDF[1])
	@inbounds for n in 2:length(EDF)
		append!(outdf, EDF[n])
	end
	return outdf
end	

elas_across_nlogit(x::Vector{T}, nl::nlogit, price_indices::Vector{Int64}) where T<:Real = elas_across_nlogit(x, nl.model, nl.data, price_indices)

# Gradient
function grad_elas_across_nlogit(x::Vector{T}, model::nlogit_model, nlnd::nlogit_case_data, price_indices::Vector{Int64}) where T<:Real
	J = 0
	for nest in nlnd 
		J += size(nest.Xj, 1)
	end
	return [ForwardDiff.gradient(θ ->elas_across_nlogit(θ,  model, nlnd, price_indices)[j], x) for j in 1:J]
end

function grad_elas_across_nlogit(x::Vector{T}, model::nlogit_model, nld::nlogit_data, price_indices::Vector{Int64}) where T<:Real
	∇e_kj = VV{eltype(x)}()
	for case_data in nld 
		append!(∇e_kj, grad_elas_across_nlogit(x, model, case_data, price_indices) )
	end
	∇e_kj
end

grad_elas_across_nlogit(x::Vector{T}, nl::nlogit, price_indices::Vector{Int64}) where T<:Real = grad_elas_across_nlogit(x, nl.model, nl.data, price_indices)


