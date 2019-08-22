
# clogit elasticities 

function elas_own_clogit(beta::Vector{T}, clcd::clogit_case_data, outside_share::Float64, price_indices::Vector{Int64}) where T<:Real
	@unpack jstar, dstar, Xj = clcd
	
	# Step 1: get price terms
	J = size(Xj,1)
	alpha_x_price = zeros(eltype(beta), J)
	for inds in price_indices
		alpha_x_price .+= Xj[:,inds].*beta[inds]
	end	

	# Step 2: Get prob purchase
	V = Xj*beta
	pr_j = (1.0 .- outside_share).*multinomial(V)

	return alpha_x_price.*(1.0 .- pr_j)
end

function elas_own_clogit(beta::Vector{T}, cld::clogit_data, outside_share::Float64, price_indices::Vector{Int64}) where T<:Real
	e_jj = eltype(beta)[]
	for case_data in cld 
		append!(e_jj, elas_own_clogit(beta, case_data , outside_share, price_indices) )
	end
	e_jj
end

elas_own_clogit(beta::Vector{T}, cl::clogit, price_indices::Vector{Int64}) where T<:Real = 
	elas_own_clogit(beta, cl.data, cl.model.opts[:outside_share], price_indices)

function grad_elas_own_clogit(beta::Vector{T}, clcd::clogit_case_data, outside_share::Float64, price_indices::Vector{Int64}) where T<:Real
	@unpack jstar, dstar, Xj = clcd
	J = size(Xj,1)
	return [ForwardDiff.gradient(b->elas_own_clogit(b , clcd, outside_share, price_indices)[j], beta) for j in 1:J]
end

function grad_elas_own_clogit(beta::Vector{T}, cld::clogit_data, outside_share::Float64, price_indices::Vector{Int64}) where T<:Real
	∇e_jj = VV{eltype(beta)}()
	for case_data in cld 
		append!(∇e_jj, grad_elas_own_clogit(beta, case_data , outside_share, price_indices) )
	end
	∇e_jj
end

grad_elas_own_clogit(beta::Vector{T}, cl::clogit, price_indices::Vector{Int64}) where T<:Real = 
	grad_elas_own_clogit(beta, cl.data, cl.model.opts[:outside_share], price_indices)


#= Cross price Elasticities =#

function elas_cross_clogit(beta::Vector{T}, clcd::clogit_case_data, outside_share::Float64, price_indices::Vector{Int64}) where T<:Real
	@unpack jstar, dstar, Xj = clcd
	
	# Step 1: get price terms
	J = size(Xj,1)
	alpha_x_price = zeros(eltype(beta), J)
	for inds in price_indices
		alpha_x_price .+= Xj[:,inds].*beta[inds]
	end	

	# Step 2: Get prob purchase
	V = Xj*beta
	pr_j = (1.0 .- outside_share).*multinomial(V)

	return -alpha_x_price.*pr_j
end

function elas_cross_clogit(beta::Vector{T}, cld::clogit_data, outside_share::Float64, price_indices::Vector{Int64}) where T<:Real
	e_kj = eltype(beta)[]
	for case_data in cld 
		append!(e_kj, elas_cross_clogit(beta, case_data , outside_share, price_indices) )
	end
	e_kj
end

elas_cross_clogit(beta::Vector{T}, cl::clogit, price_indices::Vector{Int64}) where T<:Real = 
	elas_cross_clogit(beta, cl.data, cl.model.opts[:outside_share], price_indices)


function grad_elas_cross_clogit(beta::Vector{T}, clcd::clogit_case_data, outside_share::Float64, price_indices::Vector{Int64}) where T<:Real
	@unpack jstar, dstar, Xj = clcd
	J = size(Xj,1)
	return [ForwardDiff.gradient(b->elas_cross_clogit(b , clcd, outside_share, price_indices)[j], beta) for j in 1:J]
end

function grad_elas_cross_clogit(beta::Vector{T}, cld::clogit_data, outside_share::Float64, price_indices::Vector{Int64}) where T<:Real
	∇e_kj = VV{eltype(beta)}()
	for case_data in cld 
		append!(∇e_kj, grad_elas_cross_clogit(beta, case_data , outside_share, price_indices) )
	end
	∇e_kj
end

grad_elas_cross_clogit(beta::Vector{T}, cl::clogit, price_indices::Vector{Int64}) where T<:Real = 
	grad_elas_cross_clogit(beta, cl.data, cl.model.opts[:outside_share], price_indices)

