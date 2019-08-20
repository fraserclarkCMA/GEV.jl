
# clogit elasticities 



function elas_own_clogit(beta::Vector{T}, clcd::clogit_case_data, price_indices::Vector{Int64}) where T<:Real
	@unpack jstar, dstar, Xj = clcd
	
	# Step 1: get price terms
	J = size(Xj,1)
	alpha_x_price = zeros(eltype(beta), J)
	for inds in price_indices
		alpha_x_price .+= Xj[:,inds].*beta[inds]
	end	

	# Step 2: Get prob purchase
	V = Xj*beta
	pr_j = multinomial(V)

	return alpha_x_price.*(1.0 .- pr_j)
end

function elas_own_clogit(beta::Vector{T}, cld::clogit_data, price_indices::Vector{Int64}) where T<:Real
	e_jj = eltype(beta)[]
	for case_data in cld 
		append!(e_jj, elas_own_clogit(beta, case_data , price_indices) )
	end
	e_jj
end

function grad_elas_own_clogit(beta::Vector{T}, clcd::clogit_case_data, price_indices::Vector{Int64}) where T<:Real
	@unpack jstar, dstar, Xj = clcd
	J = size(Xj,1)
	return [ForwardDiff.gradient(b->elas_own_clogit(b , clcd, price_indices)[j], beta) for j in 1:J]
end

function grad_elas_own_clogit(beta::Vector{T}, cld::clogit_data, price_indices::Vector{Int64}) where T<:Real
	∇e_jj = VV{eltype(beta)}()
	for case_data in cld 
		append!(∇e_jj, grad_elas_own_clogit(beta, case_data , price_indices) )
	end
	∇e_jj
end


#= Cross price Elasticities =#

function elas_cross_clogit(beta::Vector{T}, clcd::clogit_case_data, price_indices::Vector{Int64}) where T<:Real
	@unpack jstar, dstar, Xj = clcd
	
	# Step 1: get price terms
	J = size(Xj,1)
	alpha_x_price = zeros(eltype(beta), J)
	for inds in price_indices
		alpha_x_price .+= Xj[:,inds].*beta[inds]
	end	

	# Step 2: Get prob purchase
	V = Xj*beta
	pr_j = multinomial(V)

	return alpha_x_price.*pr_j
end

function elas_cross_clogit(beta::Vector{T}, cld::clogit_data, price_indices::Vector{Int64}) where T<:Real
	e_kj = eltype(beta)[]
	for case_data in cld 
		append!(e_kj, elas_cross_clogit(beta, case_data ,price_indices) )
	end
	e_kj
end

function grad_elas_cross_clogit(beta::Vector{T}, clcd::clogit_case_data, price_indices::Vector{Int64}) where T<:Real
	@unpack jstar, dstar, Xj = clcd
	J = size(Xj,1)
	return [ForwardDiff.gradient(b->elas_cross_clogit(b , clcd, price_indices)[j], beta) for j in 1:J]
end

function grad_elas_cross_clogit(beta::Vector{T}, cld::clogit_data, price_indices::Vector{Int64}) where T<:Real
	∇e_kj = VV{eltype(beta)}()
	for case_data in cld 
		append!(∇e_kj, grad_elas_cross_clogit(beta, case_data , price_indices) )
	end
	∇e_kj
end


#=
#= Own price Elasticities =#

function elas_own_clogit(beta::Vector{T}, X::Matrix, i::Int64, j::Int64) where T<:Real 
	pr_j = multinomial(X*beta)[j]
	return beta[i].*X[j,i].*(1.0 - pr_j)
end

function elas_own_clogit(beta::Vector{T}, clcd::clogit_case_data, varnum::Int64) where T<:Real
	@unpack jstar, dstar, Xj = clcd
	V = Xj*beta
	prob = multinomial(V)
	price = Xj[:,varnum]
	[elas_own_clogit(beta, Xj, varnum, j) for j in 1:length(V)]
end

function elas_own_clogit(beta::Vector{T}, cld::clogit_data, varnum::Int64) where T<:Real
	e_jj = eltype(beta)[]
	for case_data in cld 
		append!(e_jj, elas_own_clogit(beta, case_data ,varnum) )
	end
	e_jj
end

function grad_elas_own_clogit(beta::Vector{T}, clcd::clogit_case_data, varnum::Int64) where T<:Real
	@unpack jstar, dstar, Xj = clcd
	V = Xj*beta
	prob = multinomial(V)
	price = Xj[:,varnum]
	[ForwardDiff.gradient(b->elas_own_clogit(b, Xj, varnum, j), beta)[varnum] for j in 1:length(V)]
end

function grad_elas_own_clogit(beta::Vector{T}, cld::clogit_data, varnum::Int64) where T<:Real
	∇e_jj = eltype(beta)[]
	for case_data in cld 
		append!(∇e_jj, grad_elas_own_clogit(beta, case_data , varnum) )
	end
	∇e_jj
end

#= Cross price Elasticities =#

function elas_cross_clogit(beta::Vector{T}, X::Matrix, varnum::Int64, j::Int64) where T<:Real 
	pr_j = multinomial(X*beta)[j]
	return -beta[varnum].*X[j,varnum].*pr_j
end

function elas_cross_clogit(beta::Vector{T}, clcd::clogit_case_data, varnum::Int64) where T<:Real
	@unpack jstar, dstar, Xj = clcd
	V = Xj*beta
	prob = multinomial(V)
	price = Xj[:,varnum]
	[elas_cross_clogit(beta, Xj, varnum, j) for j in 1:length(V)]
end

function elas_cross_clogit(beta::Vector{T}, cld::clogit_data, varnum::Int64) where T<:Real
	e_jj = eltype(beta)[]
	for case_data in cld 
		append!(e_jj, elas_cross_clogit(beta, case_data ,varnum) )
	end
	e_jj
end

function grad_elas_cross_clogit(beta::Vector{T}, clcd::clogit_case_data, varnum::Int64) where T<:Real
	@unpack jstar, dstar, Xj = clcd
	V = Xj*beta
	prob = multinomial(V)
	price = Xj[:,varnum]
	[ForwardDiff.gradient(b->elas_cross_clogit(b, Xj, varnum, j), beta)[varnum] for j in 1:length(V)]
end

function grad_elas_cross_clogit(beta::Vector{T}, cld::clogit_data, varnum::Int64) where T<:Real
	∇e_jj = eltype(beta)[]
	for case_data in cld 
		append!(∇e_jj, grad_elas_cross_clogit(beta, case_data , varnum) )
	end
	∇e_jj
end
=#





