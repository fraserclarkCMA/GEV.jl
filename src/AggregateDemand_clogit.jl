#=
1. Create code computing demand derivative matrix with an existing set of choice sets and estimates

2. Add one that allows for a change of price
=#


# ADD IN DDQi here if hand code....

# New cl.data from previous commands
function DemandOutputs_clogit_case(beta::AbstractVector, clcd::clogit_case_data, 
						xvarpos::Int64, interacted_xvarpos::Union{Int64,Vector{Int64}}=[]) where T<:Real
	
	# new_xvar is already in Xj -> pass it for ForwardDiff
	@unpack case_num, jid, jstar, dstar, Xj = clcd
	
	# Step 1: get price terms
	J = size(Xj,1)
	alpha_x_price = zeros(J)
	all_xvars = vcat(xvarpos, interacted_xvarpos)
	for inds in all_xvars
		alpha_x_price .+= Xj[:,inds]*beta[inds]
	end	

	# Step 2: Get prob purchase
	V = Xj*beta
	PROBi = multinomial(V)

	# Step 3: Get individual outputs [No Hessian Yet]
	DQi = zeros(Float64,J,J)
	for (nj,j) in enumerate(jid), (nk,k) in enumerate(jid)
		if j==k
			DQi[j,k] = alpha_x_price[nj]*(1 - PROBi[nj]) * PROBi[nj] / Xj[nj,xvarpos]
		else 
			DQi[j,k] = -alpha_x_price[nj] * PROBi[nj] * PROBi[nk] / Xj[nj,xvarpos]
		end
	end 

	# Allow for choice set hetergeneity so add indicator functions
	PRODSi = zeros(Int64,J)
	PRODSi[jid] .= 1

	return (PRODS = PRODSi, CTR = PRODSi*PRODSi', PROB=PROBi, DQ = DQi)

end

# clogit input
function AggregateDemand(beta::Vector, df::DataFrame, cl::clogit, 
							xvarpos::Int64, interacted_xvarpos::Union{Int64,Vector{Int64}}=[])

	@assert eltype(df[!, cl.model.choice_id][1]) == Int64 "Enumerate choice_id to be Int64 and make new cl.model with choice_id set to this new variable"

	J = maximum(df[!, cl.model.choice_id])
	PRODS = zeros(Int64, J)
	CTR = zeros(Int64, J, J)
	PROB = zeros(Float64,J)
	DQ = zeros(Float64, J, J)
	for clcd in cl.data
		Di = DemandOutputs_clogit_case(beta, clcd, xvarpos, interacted_xvarpos)
		PRODS += Di.PRODS
		CTR += Di.CTR
		PROB += Di.PROB
		DQ += Di.DQ
	end

	return (PROB = PROB ./ PRODS, DQ = DQ ./ CTR) 

end 

function AggregateDemand(	beta::Vector, df::DataFrame, clm::clogit_model, 
				new_xvar::AbstractVector{T}, xvarname::Symbol, xvarpos::Int64, interacted_xvarpos::Union{Int64,Vector{Int64}}=[]) where T

	@assert eltype(df[!, clm.choice_id][1]) == Int64 "Enumerate choice_id to be Int64 and make new cl.model with choice_id set to this new variable"

	# Write old xvar
	df[!, Symbol(:old_,xvarname)] = df[!, xvarname]

	# Insert new xvar
	df[!, xvarname] = new_xvar[df[!,clm.choice_id]]

	# New clogit data (but same model)
	cl_new = clogit( clm, make_clogit_data(clm, df))

	return AggregateDemand(beta, df, cl_new, xvarpos, interacted_xvarpos)

end 

#=

# At existing clogit model &. data -- choice set, parameters etc. Use this version for finite difference
function AggregateElasticityMatrix(beta::AbstractVector, df::DataFrame, cl::clogit, 
				xvarname::Symbol, xvarpos::Int64, interacted_xvarpos::Union{Int64,Vector{Int64}}=[])

	@assert eltype(df[!, cl.model.choice_id][1]) == Int64 "Enumerate choice_id to be Int64 and make new cl.model with choice_id set to this new variable"

	J = maximum(df[!, cl.model.choice_id])
	prob_df = clogit_prob(beta, cl)
	rename!(prob_df, :pr_j => :prob)

	# Calculate elasticities: own, cross
	all_xvars = vcat(xvarpos, interacted_xvarpos)
	e_own_df = elas_own_clogit(beta, cl, all_xvars)
	e_cross_df = elas_cross_clogit(beta, cl, all_xvars)
	
	# Join to individual choice dataframe
	df = leftjoin(df, prob_df, on=[cl.model.case_id, cl.model.choice_id]);
	df = leftjoin(df, e_own_df, on=[cl.model.case_id, cl.model.choice_id]);
	df = leftjoin(df, e_cross_df, on=[cl.model.case_id, cl.model.choice_id]);

	# Get Aggregate Demand Derivatives Matrix
	CTR = zeros(J,J)
	E = zeros(J,J)
	for subdf in groupby(df, cl.model.case_id)
		for (nj,j) in enumerate(subdf[!,cl.model.choice_id]), (nk,k) in enumerate(subdf[!, cl.model.choice_id])
			if j==k
				E[j,k] += subdf[nj,:ejj] 
			else 
				E[j,k] += subdf[nj,:ekj] 
			end
			CTR[j,k] += 1
		end
	end
	E ./= CTR
	return E
end 

function AggregateElasticityMatrix(beta::AbstractVector, df::DataFrame, clm::clogit_model, 
				new_xvar::Vector{T}, xvarname::Symbol, xvarpos::Int64, interacted_xvarpos::Union{Int64,Vector{Int64}}=[]) where T<:Real
	
	@assert eltype(df[!, clm.choice_id][1]) == Int64 "Enumerate choice_id to be Int64 and make new cl.model with choice_id set to this new variable"

	# Write old xvar
	df[!, Symbol(:old,xvarname)] = df[!, xvarname]

	# Insert new xvar
	df[!, xvarname] = new_xvar[df[!,clm.choice_id]]

	# add cl at new data point -> mainly data

	J = maximum(df[!, clm.choice_id])
	prob_df = clogit_prob(beta, cl)
	rename!(prob_df, :pr_j => :prob)

	# Calculate elasticities: own, cross
	all_xvars = vcat(xvarpos, interacted_xvarpos)
	e_own_df = elas_own_clogit(beta, cl, all_xvars)
	e_cross_df = elas_cross_clogit(beta, cl, all_xvars)
	
	# Join to individual choice dataframe
	df = leftjoin(df, prob_df, on=[cl.model.case_id, cl.model.choice_id]);
	df = leftjoin(df, e_own_df, on=[cl.model.case_id, cl.model.choice_id]);
	df = leftjoin(df, e_cross_df, on=[cl.model.case_id, cl.model.choice_id]);

	# Get Aggregate Demand Derivatives Matrix
	CTR = zeros(eltype(new_xvar),J,J)
	E = zeros(eltype(new_xvar),J,J)
	for subdf in groupby(df, cl.model.case_id)
		for (nj,j) in enumerate(subdf[!,cl.model.choice_id]), (nk,k) in enumerate(subdf[!, cl.model.choice_id])
			if j==k
				E[j,k] += subdf[nj,:ejj] 
			else 
				E[j,k] += subdf[nj,:ekj] 
			end
			CTR[j,k] += 1
		end
	end
	E ./= CTR
	return E
end

=#
#= SLOW!!!!
# Aggregate demand derivatives at existing model , choice sets, and parameter estimates
function AggregateDemandDerivateMatrix(beta::AbstractVector, df::DataFrame, cl::clogit, 
							xvarname::Symbol, xvarpos::Int64, interacted_xvarpos::Union{Int64,Vector{Int64}}=[])

	@assert eltype(df[!, cl.model.choice_id][1]) == Int64 "Enumerate choice_id to be Int64 and make new cl.model with choice_id set to this new variable"

	J = maximum(df[!, cl.model.choice_id])
	prob_df = clogit_prob(beta, cl)
	rename!(prob_df, :pr_j => :prob)

	# Calculate elasticities: own, cross
	all_xvars = vcat(xvarpos, interacted_xvarpos)
	e_own_df = elas_own_clogit(beta, cl, all_xvars)
	e_cross_df = elas_cross_clogit(beta, cl, all_xvars)
	
	# Join to individual choice dataframe
	df = leftjoin(df, prob_df, on=[cl.model.case_id, cl.model.choice_id]);
	df = leftjoin(df, e_own_df, on=[cl.model.case_id, cl.model.choice_id]);
	df = leftjoin(df, e_cross_df, on=[cl.model.case_id, cl.model.choice_id]);

	# Get Aggregate Demand Derivatives Matrix
	CTR = zeros(J,J)
	DQ = zeros(J,J)
	for subdf in groupby(df, cl.model.case_id)
		for (nj,j) in enumerate(subdf[!,cl.model.choice_id]), (nk,k) in enumerate(subdf[!, cl.model.choice_id])
			if j==k
				DQ[j,k] += subdf[nj,:ejj] * subdf[nj,:prob] / subdf[nj,xvarname]
			else 
				DQ[j,k] += subdf[nj,:ekj] * subdf[nk,:prob] / subdf[nj,xvarname]
			end
			CTR[j,k] += 1
		end
	end
	DQ ./= CTR
	return DQ
end 


# Aggregate demand derivatives at new xvariable
function AggregateDemandDerivateMatrix(beta::AbstractVector, df::DataFrame, clm::clogit_model, new_xvar::Vector{T}, xvarname::Symbol, xvarpos::Vector, interacted_xvarpos::Vector=[]) where T<:Real

	@assert eltype(df[!, clm.choice_id][1]) == Int64 "Enumerate choice_id to be Int64 and make new cl.model with choice_id set to this new variable"

	# Write old xvar
	df[!, Symbol(:old,xvarname)] = df[!, xvarname]

	# Insert new xvar
	df[!, xvarname] = new_xvar[df[!,clm.choice_id]]

	# Need to remake cl logit demand model -> cl2 to remake choice sets
	J = maximum(df[!, clm.choice_id])

	# New clogit data (but same model)
	cl2 = clogit( clm, make_clogit_data(clm, df))
	prob_df = clogit_prob(beta, cl2)
	rename!(prob_df, :pr_j => :prob)

	# Calculate elasticities: own, cross
	all_xvars = vcat(xvarpos, interacted_xvarpos)
	e_own_df = elas_own_clogit(beta, cl2, all_xvars)
	e_cross_df = elas_cross_clogit(beta, cl2, all_xvars)
	
	# Join to individual choice dataframe
	df = leftjoin(df, prob_df, on=[clm.case_id, clm.choice_id]);
	df = leftjoin(df, e_own_df, on=[clm.case_id, clm.choice_id]);
	df = leftjoin(df, e_cross_df, on=[clm.case_id, clm.choice_id]);

	# Get Aggregate Demand Derivatives Matrix
	CTR = zeros(eltype(new_xvar),J,J)
	DQ = zeros(eltype(new_xvar),J,J)
	for subdf in groupby(df, clm.case_id)
		for (nj,j) in enumerate(subdf[!,clm.choice_id]), (nk,k) in enumerate(subdf[!, clm.choice_id])
			if j==k
				DQ[j,k] += subdf[nj,:ejj] * subdf[nj,:prob] / subdf[nj,xvarname]
			else 
				DQ[j,k] += subdf[nj,:ekj] * subdf[nk,:prob] / subdf[nj,xvarname]
			end
			CTR[j,k] += 1
		end
	end
	DQ ./= CTR
	return DQ
end 
=#

#=
function AggregateDemandDerivateMatrix(new_xvar::AbstractVector{T}, xvarname::Symbol, 
				beta::Vector, df::DataFrame, cl::clogit, xvarpos::Vector, interacted_xvarpos::Vector=[]) where T

	@assert eltype(df[!, cl.model.choice_id][1]) == Int64 "Enumerate choice_id to be Int64 and make new cl.model with choice_id set to this new variable"

	# Write old xvar
	df[!, Symbol(:old,xvarname)] = df[!, xvarname]

	# Insert new xvar
	df[!, xvarname] = new_xvar[df[!,cl.model.choice_id]]

	return AggregateDemandDerivateMatrix(xvarname, beta, df, cl, xvarpos, interacted_xvarpos)

end 
=#

