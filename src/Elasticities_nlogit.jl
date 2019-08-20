
function elas_own_nlogit(x::Vector{T}, θ::nlogit_param, nlnd::nlogit_case_data, flags::Dict, idx::Dict, RUM::Bool, outside_share::Float64, price_indices::Vector{Int64}) where T<:Real
	
	vec_to_theta!(x, θ, flags, idx)	
	@unpack beta, alpha, lambda = θ

	Numλ = length(lambda)	

	pr_j, pr_jg = nlogit_prob(x, θ, nlnd, flags, idx, RUM, outside_share)
	pr_g = pr_j./pr_jg

	e_jj = eltype(x)[]
	ctr = 0
	for nest_data in nlnd 
		@unpack jstar, dstar, nest_star, nest_num, Xj, Wk = nest_data
		J = size(Xj,1)

		# Step 1: Price Terms
		alpha_x_price = zeros(eltype(x),J)
		for inds in price_indices
			alpha_x_price .+= Xj[:,inds].*beta[inds]
		end			
			
		# Step 2: Get nest parameter
		λg = Numλ > 1 ? lambda[nest_num] : lambda[1]

		# Step 3: Calculate elasticity
		own_elas = zeros(eltype(x),J)
		for j in eachindex(own_elas)
			ctr += 1
			own_elas[j] = alpha_x_price[j].*( 1.0 .- λg.*pr_j[ctr] .- (1.0 .- λg).*pr_jg[ctr] ) ./λg
		end

		# Step 4: Append 
		append!(e_jj, own_elas)

	end

	return e_jj

end

elas_own_nlogit(x::Vector{T}, model::nlogit_model, nlnd::nlogit_case_data, price_indices::Vector{Int64}) where T<:Real =
	elas_own_nlogit(x, model.params, nlnd, model.flags, model.idx, model.opts[:RUM], model.opts[:outside_share], price_indices) 

function elas_own_nlogit(x::Vector{T}, model::nlogit_model, nld::nlogit_data, price_indices::Vector{Int64}) where T<:Real
	e_jj = eltype(x)[]
	for case_data in nld
		append!(e_jj, elas_own_nlogit(x, model, case_data, price_indices) )
	end
	return e_jj
end	

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


function elas_within_nlogit(x::Vector{T}, θ::nlogit_param, nlnd::nlogit_case_data, flags::Dict, idx::Dict, RUM::Bool, outside_share::Float64, price_indices::Vector{Int64}) where T<:Real
	
	vec_to_theta!(x, θ, flags, idx)	
	@unpack beta, alpha, lambda = θ

	Numλ = length(lambda)	

	pr_j, pr_jg = nlogit_prob(x, θ, nlnd, flags, idx, RUM, outside_share)
	pr_g = pr_j./pr_jg

	e_kj = eltype(x)[]
	ctr = 0
	for nest_data in nlnd 
		@unpack jstar, dstar, nest_star, nest_num, Xj, Wk = nest_data
		J = size(Xj,1)

		# Step 1: Price Terms
		alpha_x_price = zeros(eltype(x),J)
		for inds in price_indices
			alpha_x_price .+= Xj[:,inds].*beta[inds]
		end			
			
		# Step 2: Get nest parameter
		λg = Numλ > 1 ? lambda[nest_num] : lambda[1]

		# Step 3: Calculate elasticity
		within_elas = zeros(eltype(x),J)
		for j in eachindex(within_elas)
			ctr += 1
			within_elas[j] = -alpha_x_price[j].*( λg.*pr_j[ctr] .+ (1.0 .- λg).*pr_jg[ctr] ) ./λg
		end

		# Step 4: Append 
		append!(e_kj, within_elas)

	end

	return e_kj

end

elas_within_nlogit(x::Vector{T}, model::nlogit_model, nlnd::nlogit_case_data, price_indices::Vector{Int64}) where T<:Real =
	elas_within_nlogit(x, model.params, nlnd, model.flags, model.idx, model.opts[:RUM], model.opts[:outside_share], price_indices) 

function elas_within_nlogit(x::Vector{T}, model::nlogit_model, nld::nlogit_data, price_indices::Vector{Int64}) where T<:Real
	e_kj = eltype(x)[]
	for case_data in nld
		append!(e_kj, elas_within_nlogit(x, model, case_data, price_indices) )
	end
	return e_kj
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

function elas_across_nlogit(x::Vector{T}, θ::nlogit_param, nlnd::nlogit_case_data, flags::Dict, idx::Dict, RUM::Bool, outside_share::Float64, price_indices::Vector{Int64}) where T<:Real
	
	vec_to_theta!(x, θ, flags, idx)	
	@unpack beta, alpha, lambda = θ

	Numλ = length(lambda)	

	pr_j, pr_jg = nlogit_prob(x, θ, nlnd, flags, idx, RUM, outside_share)

	e_kj = eltype(x)[]
	ctr = 0
	for nest_data in nlnd 
		@unpack jstar, dstar, nest_star, nest_num, Xj, Wk = nest_data
		J = size(Xj,1)

		# Step 1: Price Terms
		alpha_x_price = zeros(eltype(x),J)
		for inds in price_indices
			alpha_x_price .+= Xj[:,inds].*beta[inds]
		end			
			
		# Step 2: Get nest parameter
		λg = Numλ > 1 ? lambda[nest_num] : lambda[1]

		# Step 3: Calculate elasticity
		across_elas = zeros(eltype(x),J)
		for j in eachindex(across_elas)
			ctr += 1
			across_elas[j] = -alpha_x_price[j].*pr_j[ctr]
		end

		# Step 4: Append 
		append!(e_kj, across_elas)

	end

	return e_kj

end

elas_across_nlogit(x::Vector{T}, model::nlogit_model, nlnd::nlogit_case_data, price_indices::Vector{Int64}) where T<:Real =
	elas_across_nlogit(x, model.params, nlnd, model.flags, model.idx, model.opts[:RUM], model.opts[:outside_share], price_indices) 

function elas_across_nlogit(x::Vector{T}, model::nlogit_model, nld::nlogit_data, price_indices::Vector{Int64}) where T<:Real
	e_kj = eltype(x)[]
	for case_data in nld
		append!(e_kj, elas_across_nlogit(x, model, case_data, price_indices) )
	end
	return e_kj
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


