#=
1. Create code computing demand derivative matrix with an existing set of choice sets and estimates

2. Add one that allows for a change of price
=#

# New cl.data from previous commands
function DemandOutputs_clogit_case(beta::Vector, clcd::clogit_case_data, J::Int64,
						xvarpos::Int64, interacted_xvarpos::ScalarOrVector{Int64}=Int64[],)
	
	@unpack case_num, jid, jstar, dstar, Xj, xlevel_id, xlevel_var = clcd
	
	if length(xlevel_var)==0
		xlevel_var = Xj[:, xvarpos]
	end

	# Step 1: get price terms
	J_CSi = size(Xj,1)
	alpha_x_xvar = zeros(J_CSi)
	all_xvars = vcat(xvarpos, interacted_xvarpos)
	for inds in all_xvars
		alpha_x_xvar .+= Xj[:,inds]*beta[inds]
	end	

	# Step 2: Get prob purchase
	V = Xj*beta
	PROBi = multinomial(V)


	temp_own = alpha_x_xvar .* (1 .- PROBi) .* PROBi ./ xlevel_var
	temp_cross = -alpha_x_xvar .* PROBi ./ xlevel_var
	temp_own[isnan.(temp_own)] .= 0
	temp_cross[isnan.(temp_cross)] .= 0

	# Step 3: Get individual outputs
	DQi = zeros(J,J)
	@inbounds for (nj,j) in enumerate(jid), (nk,k) in enumerate(jid)
		if j==k
			DQi[j,k] = temp_own[nj]
		else 
			DQi[j,k] = temp_cross[nj] * PROBi[nk]
		end
	end 


#=
	# Step 3: Get individual outputs
	DQi = zeros(J,J)
	@inbounds for (nj,j) in enumerate(jid), (nk,k) in enumerate(jid)
		if xlevel_var[nj]==0 
			nothing
		elseif j==k
			DQi[j,k] = alpha_x_xvar[nj] .* (1 .- PROBi[nj]) .* PROBi[nj] ./ xlevel_var[nj]
		else 
			DQi[j,k] = -alpha_x_xvar[nj] .* PROBi[nj] .* PROBi[nk] / xlevel_var[nj]
		end
	end 
=#
	# Allow for choice set hetergeneity so add indicator functions
	PRODSi = zeros(Int64,J)
	PRODSi[jid] .= 1
	PROB = zeros(Float64,J)
	PROB[jid] .= PROBi

	return (PRODS = PRODSi, CTR = PRODSi*PRODSi', PROB=PROB, DQ = DQi , CW = logsumexp(V))

end

# Loop over all choice sets
function AggregateDemand(beta::Vector, df::DataFrame, cl::clogit,
							xvarpos::Int64, interacted_xvarpos::ScalarOrVector{Int64}=Int64[])

	@assert eltype(df[!, cl.model.choice_id][1]) == Int64 "Enumerate choice_id to be Int64 and make new cl.model with choice_id set to this new variable"

	J = maximum(df[!, cl.model.choice_id])
	CW = 0.
	PRODS = zeros(Int64, J)
	CTR = zeros(Int64, J, J)
	PROB = zeros(Float64,J)
	DQ = zeros(Float64, J, J)
	@inbounds for clcd in cl.data
		Di = DemandOutputs_clogit_case(beta, clcd, J, xvarpos, interacted_xvarpos)
		PRODS += Di.PRODS
		CTR += Di.CTR
		PROB += Di.PROB
		DQ += Di.DQ
		CW += Di.CW
	end

	# In case product does not show up in choice set
	PROB = PROB ./ PRODS
	DQ = DQ ./ CTR
	DQ[isnan.(DQ)] .= 0.
	PROB[isnan.(PROB)] .= 0.

	return (PROB = PROB, DQ = DQ , CW = CW) 

end 

# New point
function AggregateDemand(	beta::Vector, df::DataFrame, clm::clogit_model, 
								new_xvar::Vector{Float64}, xvarname::Symbol, 
									xvarpos::Int64, interacted_xvarpos::ScalarOrVector{Int64}=Int64[])

	@assert eltype(df[!, clm.choice_id][1]) == Int64 "Enumerate choice_id to be Int64 and make new cl.model with choice_id set to this new variable"

	df[!, Symbol(:old_,xvarname)] = df[!, xvarname]
	
	# Insert new xvar
	df[!, xvarname] = new_xvar[df[!,clm.choice_id]]

	# New clogit data (but same model)
	cl_new = clogit( clm, make_clogit_data(clm, df))

	return AggregateDemand(beta, df, cl_new, xvarpos, interacted_xvarpos)

end 

# New point special point
function AggregateDemand(beta::Vector, df::DataFrame, clm::clogit_model, 
							new_xvar_level::Vector{Float64}, individual_varname::Symbol, xvarname::Symbol,
								xvarpos::Int64, interacted_xvarpos::ScalarOrVector{Int64}=Int64[])

	@assert eltype(df[!, clm.choice_id][1]) == Int64 "Enumerate choice_id to be Int64 and make new cl.model with choice_id set to this new variable"

	# Old level
	df[!, Symbol(:old_,clm.opts[:xlevel_id])] = df[!, clm.opts[:xlevel_id]]
	
	# Old interaction
	df[!, Symbol(:old_,xvarname)] = deepcopy(df[!, xvarname])

	# New Level
	df[!, clm.opts[:xlevel_id]] = new_xvar_level[df[!,clm.choice_id]]

	# New Level with old interaction
	df[!, xvarname] = df[!, clm.opts[:xlevel_id]] .* df[!, individual_varname];


	# New clogit data (but same model)
	cl_new = clogit( clm, make_clogit_data(clm, df))

	return AggregateDemand(beta, df, cl_new, xvarpos, interacted_xvarpos)

end 



#=
# Loop over all choice sets
function AggregateDemand(	beta::Vector, df::DataFrame, clm::clogit_model, 
								new_xvar::Vector{Float64}, xvarname::Symbol, 
									xvarpos::Int64, interacted_xvarpos::ScalarOrVector{Int64}=Int64[])

	@assert eltype(df[!, clm.choice_id][1]) == Int64 "Enumerate choice_id to be Int64 and make new cl.model with choice_id set to this new variable"

	# Write old xvar
	df[!, Symbol(:old_,xvarname)] = df[!, xvarname]

	# Insert new xvar
	df[!, xvarname] = new_xvar[df[!,clm.choice_id]]

	# New clogit data (but same model)
	cl_new = clogit( clm, make_clogit_data(clm, df))

	return AggregateDemand(beta, df, cl_new, new_xvar, xvarpos, interacted_xvarpos)

end 
=#
# Aggregate Elasticities and Diversion Ratios
function AggregateDiversionRatioMatrix( DQ::Matrix , INDMAT::Matrix)
	(N,J) = size(INDMAT)
	dsj_dpj = diag(DQ)
	DR = zeros(N, N)
	for a in 1:N
		dsAdpA = sum(dsj_dpj .* INDMAT[a, :])
		for b in 1:N 
			if a!==b
				dsBdpA = sum(DQ.*(INDMAT[a,:]*INDMAT[b,:]'))
				DR[a,b] = - dsBdpA / dsAdpA 
			end
		end 
	end
	return DR
end 

function AggregateElasticityMatrix( DQ::Matrix, prob::Vector, xvar::Vector, INDMAT::Matrix)

	(N,J) = size(INDMAT)
	dsj_dxj = diag(DQ)

	E = zeros(N, N)
	for a in 1:N # xA up
		sj_in_A = prob .* INDMAT[a,:]
		sA = sum(sj_in_A) 										
		xA = sum(xvar .* (sj_in_A / sA)) 						
		E[a,a] = sum(dsj_dxj .* INDMAT[a, :]) * xA / sA
		for b in 1:N  # sB responds
			if a!==b 
				sj_in_B = prob .* INDMAT[b, :] 
				sB = sum(sj_in_B)
				E[a,b] = sum(DQ.*(INDMAT[a,:]*INDMAT[b,:]')) * xA / sB
			end
		end 
	end

	return E

end 
