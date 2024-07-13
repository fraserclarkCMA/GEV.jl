
# clogit elasticities 

function elas_own_clogit(beta::Vector{T}, clcd::clogit_case_data, outside_share::Float64, 
							case_id::Symbol, choice_id::Symbol, xvar_indices::Vector{Int64}) where T<:Real
	@unpack case_num, jid, jstar, dstar, Xj, xlevel_id, xlevel_var = clcd
	
	# Step 1: get price terms
	J = size(Xj,1)
	alpha_x_xvar = zeros(eltype(beta), J)
	for inds in xvar_indices
		alpha_x_xvar .+= Xj[:,inds].*beta[inds]
	end	

	# Step 2: Get prob purchase
	V = Xj*beta
	#pr_j = (1.0 .- outside_share).*multinomial(V)
	pr_j = multinomial(V)

	return DataFrame(case_id=>case_num.*ones(J), choice_id=>jid, :ejj=>alpha_x_xvar.*(1.0 .- pr_j)) 
end

function elas_own_clogit(beta::Vector{T}, cld::clogit_data, outside_share::Float64, 
							case_id::Symbol, choice_id::Symbol, xvar_indices::Vector{Int64}) where T<:Real
	
	EDF = DataFrame[]
	for case_data in cld
		push!(EDF, elas_own_clogit(beta, case_data , outside_share, case_id, choice_id, xvar_indices))
	end
	outdf = deepcopy(EDF[1])
	@inbounds for n in 2:length(EDF)
		append!(outdf, EDF[n])
	end
	return outdf
end

elas_own_clogit(beta::Vector{T}, cl::clogit, xvar_indices::Vector{Int64}) where T<:Real = 
	elas_own_clogit(beta, cl.data, cl.model.opts[:outside_share], cl.model.case_id, cl.model.choice_id, xvar_indices)

	
#= Cross price Elasticities =#

function elas_cross_clogit(beta::Vector{T}, clcd::clogit_case_data, outside_share::Float64,
							case_id::Symbol, choice_id::Symbol, xvar_indices::Vector{Int64}) where T<:Real
	@unpack case_num, jid, jstar, dstar, Xj, xlevel_id, xlevel_var = clcd
	small = eps()

	# Step 1: get price terms
	J = size(Xj,1)
	alpha_x_xvar = zeros(eltype(beta), J)
	for inds in xvar_indices
		alpha_x_xvar .+= Xj[:,inds].*beta[inds]
	end	

	# Step 2: Get prob purchase
	V = Xj*beta
	pr_j = (1.0 .- outside_share).*max.(small, multinomial(V))

	return DataFrame(case_id=>case_num.*ones(J), choice_id=>jid, :ekj=>-alpha_x_xvar.*pr_j) 
end

function elas_cross_clogit(beta::Vector{T}, cld::clogit_data, outside_share::Float64, 
								case_id::Symbol, choice_id::Symbol, xvar_indices::Vector{Int64}) where T<:Real
	
	EDF = DataFrame[]
	for case_data in cld
		push!(EDF, elas_cross_clogit(beta, case_data , outside_share, case_id, choice_id, xvar_indices))
	end
	outdf = deepcopy(EDF[1])
	@inbounds for n in 2:length(EDF)
		append!(outdf, EDF[n])
	end
	return outdf
end

elas_cross_clogit(beta::Vector{T}, cl::clogit, xvar_indices::Vector{Int64}) where T<:Real = 
	elas_cross_clogit(beta, cl.data, cl.model.opts[:outside_share], cl.model.case_id, cl.model.choice_id, xvar_indices)

