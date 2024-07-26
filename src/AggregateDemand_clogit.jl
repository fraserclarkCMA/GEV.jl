
function new_clogit_data(df::DataFrame, clm::clogit_model, x::Vector{Float64}, xvar::Symbol)
	
	# Copy old xvar - dataframe modified in place
	df[!, Symbol(:old_,xvar)] = df[!, xvar]
	# Insert new xvar - dataframe modified in place
	df[!, xvar] = x[df[!,clm.choice_id]]
	# New clogit data (but same model)
	return clogit( clm, make_clogit_data(clm, df))

end 

# New point special point
function new_clogit_data(df::DataFrame, clm::clogit_model, p::Vector{Float64}, xvar::Symbol, pvar::Symbol, zvar::Symbol)
	
	# Old Price
	df[!, Symbol(:old_,pvar)] = df[!, pvar]
	# Old xvar
	df[!, Symbol(:old_,xvar)] = deepcopy(df[!, xvar])
	# New x 
	df[!, pvar] = p[df[!,clm.choice_id]]
	# New Level with interaction
	df[!, xvar] = df[!, pvar] .* df[!, zvar];
	return clogit( clm, make_clogit_data(clm, df))

end 

# Get individual demand output from their choice set
function DemandOutputs_clogit_case(beta::Vector, clcd::clogit_case_data, xvarpos::Int64)
	
	@unpack case_num, jid, jstar, dstar, Xj, pvar, p, zvar, z = clcd
	
	jid = convert.(Int64, jid)

	# Step 1: get price terms
	J = size(Xj,1)
	alpha_x_xvar = zeros(J)
	alpha_x_xvar .+= Xj[:,xvarpos]*beta[xvarpos]

	# Step 2: Get prob purchase
	V = Xj*beta
	s = multinomial(V)
	cw = logsumexp(V)

	temp_own = alpha_x_xvar .* (1 .- s) .* s  ./ Xj[:, xvarpos]
	temp_cross = -alpha_x_xvar .* s ./ Xj[:, xvarpos]
	dsdx = (repeat(temp_cross, 1, J) .* repeat(s, 1, J)') .* (1 .- I(J)) .+ diagm(temp_own)

	if pvar == zvar == Symbol()
		return clogit_case_output(case_num, J, jid, Xj[:, xvarpos], s, dsdx, cw, Float64[], Float64[])
	elseif  zvar == Symbol()
		return clogit_case_output(case_num, J, jid, Xj[:, xvarpos], s, dsdx, cw, p, Float64[])
	else 
		return clogit_case_output(case_num, J, jid, Xj[:, xvarpos], s, dsdx, cw, p, z)
	end 
end


# Get individual demand output from their choice set with interactions in xvar passed in xvarpos
function DemandOutputs_clogit_case(beta::Vector, clcd::clogit_case_data, xvarpos::Int64, xzvarpos::Int64)
	
	@unpack case_num, jid, jstar, dstar, Xj, pvar, p, zvar, z = clcd
	
	jid = convert.(Int64, jid)

	# Step 1: get price terms
	J = size(Xj,1)
	alpha_x_xvar = zeros(J)
	allvars = vcat(xvarpos, xzvarpos)
	for inds in allvars
		alpha_x_xvar .+= Xj[:,inds]*beta[inds]
	end	

	# Step 2: Get prob purchase
	V = Xj*beta
	s = multinomial(V)
	cw = logsumexp(V)

	temp_own = alpha_x_xvar .* (1 .- s) .* s  ./ Xj[:, xvarpos]
	temp_cross = -alpha_x_xvar .* s ./ Xj[:, xvarpos]
	dsdx = (repeat(temp_cross, 1, J) .* repeat(s, 1, J)') .* (1 .- I(J)) .+ diagm(temp_own)

	if pvar == zvar == Symbol()
		return clogit_case_output(case_num, J, jid, Xj[:, xvarpos], s, dsdx, cw, Float64[], Float64[])
	elseif  zvar == Symbol()
		return clogit_case_output(case_num, J, jid, Xj[:, xvarpos], s, dsdx, cw, p, Float64[])
	else 
		return clogit_case_output(case_num, J, jid, Xj[:, xvarpos], s, dsdx, cw, p, z)
	end 
end

# Get individual demand output from their choice set with interactions in xvar passed in xvarpos
function DemandOutputs_clogit_case(beta::Vector, clcd::clogit_case_data, xvarpos::Int64, xzvarpos::Vector{Int64})
	
	@unpack case_num, jid, jstar, dstar, Xj, pvar, p, zvar, z = clcd
	
	jid = convert.(Int64, jid)

	# Step 1: get price terms
	J = size(Xj,1)
	alpha_x_xvar = zeros(J)
	allvars = vcat(xvarpos, xzvarpos)
	for inds in allvars
		alpha_x_xvar .+= Xj[:,inds]*beta[inds]
	end	

	# Step 2: Get prob purchase
	V = Xj*beta
	s = multinomial(V)
	cw = logsumexp(V)

	temp_own = alpha_x_xvar .* (1 .- s) .* s  ./ Xj[:, xvarpos]
	temp_cross = -alpha_x_xvar .* s ./ Xj[:, xvarpos]
	dsdx = (repeat(temp_cross, 1, J) .* repeat(s, 1, J)') .* (1 .- I(J)) .+ diagm(temp_own)

	if pvar == zvar == Symbol()
		return clogit_case_output(case_num, J, jid, Xj[:, xvarpos], s, dsdx, cw, Float64[], Float64[])
	elseif  zvar == Symbol()
		return clogit_case_output(case_num, J, jid, Xj[:, xvarpos], s, dsdx, cw, p, Float64[])
	else 
		return clogit_case_output(case_num, J, jid, Xj[:, xvarpos], s, dsdx, cw, p, z)
	end 
end


# Loop over all individuals to create aggregate demand compononts
function AggregateDemand(beta::Vector, df::DataFrame, cl::clogit, xvarpos::Int64, parallel::Bool=false)

	@assert eltype(df[!, cl.model.choice_id][1]) == Int64 "Enumerate choice_id to be Int64 and make new cl.model with choice_id set to this new variable"

	AD = parallel ?
		pmap(clcd->GEV.DemandOutputs_clogit_case(beta, clcd, xvarpos), cl.data) : 
		map(clcd->GEV.DemandOutputs_clogit_case(beta, clcd, xvarpos), cl.data);

	return AD

end

# Loop over all individuals to create aggregate demand compononts
function AggregateDemand(beta::Vector, df::DataFrame, cl::clogit, xvarpos::Int64, xzvarpos::Int64, parallel::Bool=false)

	@assert eltype(df[!, cl.model.choice_id][1]) == Int64 "Enumerate choice_id to be Int64 and make new cl.model with choice_id set to this new variable"

	AD = parallel ?
		pmap(clcd->GEV.DemandOutputs_clogit_case(beta, clcd, xvarpos, xzvarpos), cl.data) : 
		map(clcd->GEV.DemandOutputs_clogit_case(beta, clcd, xvarpos, xzvarpos), cl.data);

	return AD

end

# Loop over all individuals to create aggregate demand compononts
function AggregateDemand(beta::Vector, df::DataFrame, cl::clogit, xvarpos::Int64, xzvarpos::Vector{Int64}, parallel::Bool=false)

	@assert eltype(df[!, cl.model.choice_id][1]) == Int64 "Enumerate choice_id to be Int64 and make new cl.model with choice_id set to this new variable"

	AD = parallel ?
		pmap(clcd->GEV.DemandOutputs_clogit_case(beta, clcd, xvarpos, xzvarpos), cl.data) : 
		map(clcd->GEV.DemandOutputs_clogit_case(beta, clcd, xvarpos, xzvarpos), cl.data);

	return AD

end


# ALL OF MY SUM FUNCTIONS NEED TO BE SPARSE + SPOWNERSHIP MATRIX

# Get aggregate consumer welfare
getCW(AD::Vector{clogit_case_output}) = sum(ad.cw for ad in AD)

# ---------------------------------------- #
# Getters for product level demand outputs
# ---------------------------------------- #


getX(AD::Vector{clogit_case_output}) = sum(ad.s .* ad.x for ad in AD) ./ sum(ad.s for ad in AD)

getP(AD::Vector{clogit_case_output}) = sum(ad.s .* ad.p for ad in AD) ./ sum(ad.s for ad in AD)

getQty(AD::Vector{clogit_case_output}, M::Real=1) = sum(ad.s for ad in AD)

function getShares(AD::Vector{clogit_case_output}) 
	qj = getQty(AD)
	return qj ./ sum(qj)
end

getdQdX(AD::Vector{clogit_case_output}) = sum(ad.dsdx for ad in AD)

getdQdP(AD::Vector{clogit_case_output}, PdivY::Bool=false) = PdivY ? sum(ad.dsdx .* ad.z for ad in AD) : sum(ad.dsdx for ad in AD)

function getDiversionRatioMatrix(AD::Vector{clogit_case_output}, PdivY::Bool=false)
	dQdP = getdQdP(AD, PdivY)
	return -dQdP ./ diag(dQdP)
end 

function getElasticityMatrix(AD::Vector{clogit_case_output}, PdivY::Bool=false) 
	if PdivY 
		Q = getQty(AD)
		P = getP(AD)	
		dQdP = getdQdP(AD, true)
		return dQdP.* P ./ Q' 
	else 
		Q = getQty(AD)
		X = getX(AD)	
		dQdX = getdQdP(AD, false)
		return dQdX.* X ./ Q' 
	end
end

# ---------------------------------------- #
# Sparse Getters for product level demand outputs
# ---------------------------------------- #

function spgetX(AD::Vector{clogit_case_output})
	num = sum(sparsevec(ad.jid, ad.s .* ad.x) for ad in AD) 
	denom = sum(sparsevec(ad.jid,ad.s) for ad in AD)
	return sparsevec(denom.nzind, num.nzval ./ denom.nzval)
end

function spgetP(AD::Vector{clogit_case_output}) 
	num = sum(sparsevec(ad.jid, ad.s .* ad.p) for ad in AD) 
	denom = sum(sparsevec(ad.jid,ad.s) for ad in AD)
	return sparsevec(denom.nzind, num.nzval ./ denom.nzval)
end

spgetQty(AD::Vector{clogit_case_output}, M::Real=1) = sum(sparsevec(ad.jid, ad.s) for ad in AD)

function spgetShares(AD::Vector{clogit_case_output}) 
	qj = spgetQty(AD)
	return qj ./ sum(qj)
end

function spgetdQdX(AD::Vector{clogit_case_output}) 
	J = maximum(maximum(ad.jid) for ad in AD)
	return  sum(sparse(repeat(ad.jid, 1, ad.J)[:], repeat(ad.jid, 1, ad.J)'[:], ad.dsdx[:], J, J) for ad in AD)
end

function spgetdQdP(AD::Vector{clogit_case_output}, PdivY::Bool=false)
	J = maximum(maximum(ad.jid) for ad in AD)
	if PdivY 
		return sum( sparse( repeat(ad.jid, 1, ad.J)[:], repeat(ad.jid, 1, ad.J)'[:] , (ad.z .* ad.dsdx)[:], J, J) for ad in AD)
	else 
		return spgetdQdX(AD)
	end		
end

function spgetDiversionRatioMatrix(AD::Vector{clogit_case_output}, PdivY::Bool=false)
	dQdP = spgetdQdP(AD)
	(J, _) = size(dQdP)
	# Do this in a loop? 
	DR = spzeros(J,J)
	rows = rowvals(dQdP)
	for j in unique(rows)
		dQdpj = dQdP[j,:]
		DR[j, dQdpj.nzind] = (- dQdpj / dQdP[j,j]).nzval
	end
	return DR
end 

function spgetElasticityMatrix(AD::Vector{clogit_case_output}, PdivY::Bool=false) 
	if PdivY 
		Q = spgetQty(AD)
		P = spgetP(AD)	
		dQdP = spgetdQdP(AD, true)
		I = findnz(dQdP)
		return sparse(I[1], I[2],  dQdP.nzval .* (P.nzval ./ Q.nzval')[:]) 
	else 
		Q = spgetQty(AD)
		X = spgetX(AD)	
		dQdX = spgetdQdP(AD, false)
		I = findnz(dQdX)
		return sparse(I[1], I[2],  dQdX.nzval .* (X.nzval ./ Q.nzval')[:]) 
	end
end

# ------------------------------------------- #
# Getters for grouped aggregate demand outputs
# ------------------------------------------- #

# INDMAT is a G x J matrix whose [g,j]-th entry is 1 if j belongs to group g and 0 otherwise

function getGroupX(AD::Vector{clogit_case_output}, INDMAT::Matrix)
	grp_q = INDMAT*sum(ad.s for ad in AD)
	grp_qx = INDMAT*sum(ad.s .* ad.x for ad in AD)	
	return grp_qx ./ grp_q	
end

function getGroupP(AD::Vector{clogit_case_output}, INDMAT::Matrix)
	grp_q = INDMAT*sum(ad.s for ad in AD)
	grp_rev = INDMAT*sum(ad.s .* ad.p for ad in AD)	
	return grp_rev ./ grp_q	
end

getGroupQty(AD::Vector{clogit_case_output}, INDMAT::Matrix, M::Real=1) = INDMAT*getQty(AD, M)

getGroupShares(AD::Vector{clogit_case_output}, INDMAT::Matrix) = INDMAT*getShares(AD)

function getGroupdQdP(AD::Vector{clogit_case_output}, INDMAT::Matrix, PdivY::Bool=false)  
	(G,J) = size(INDMAT)
	dQdP = getdQdP(AD, PdivY)
	dQjdPj = diag(dQdP)
	grp_dQdP = zeros(G, G)
	@inbounds for a in 1:G
		grp_dQdP[a,a] = sum(dQjdPj .* INDMAT[a, :])
		for b in 1:G
			if a!==b
				grp_dQdP[a,b] = sum(dQdP.*(INDMAT[a,:]*INDMAT[b,:]'))
			end
		end 
	end
	return grp_dQdP
end 

function getGroupDiversionRatioMatrix(AD::Vector{clogit_case_output}, INDMAT::Matrix, PdivY::Bool=false)  
	(G,J) = size(INDMAT)
	dQdP = getdQdP(AD, PdivY)
	dQjdPj = diag(dQdP)
	DR = zeros(G, G)
	@inbounds for a in 1:G
		dQAdPA = sum(dQjdPj .* INDMAT[a, :])
		for b in 1:G
			if a!==b
				dQBdPA = sum(dQdP.*(INDMAT[a,:]*INDMAT[b,:]'))
				DR[a,b] = - dQBdPA / dQAdPA 
			end
		end 
	end
	return DR
end 

function getGroupElasticityMatrix(AD::Vector{clogit_case_output}, INDMAT::Matrix, PdivY::Bool=false)
	if PdivY 
		Q = getGroupQty(AD, INDMAT)
		P = getGroupP(AD, INDMAT)	
		dQdP = getGroupdQdP(AD, INDMAT, true)
		return dQdP.* P ./ Q' 
	else 
		Q = getGroupQty(AD, INDMAT)
		X = getGroupX(AD, INDMAT)	
		dQdX = getGroupdQdP(AD, INDMAT)
		return dQdX .* X ./ Q' 
	end
end


# ------------------------------------------- #
# Sparse Getters for grouped aggregate demand outputs
# ------------------------------------------- #

# INDMAT is a G x J matrix whose [g,j]-th entry is 1 if j belongs to group g and 0 otherwise

function spgetGroupX(AD::Vector{clogit_case_output}, INDMAT::SparseMatrixCSC)
	num = sum(sparsevec(ad.jid, ad.s .* ad.x) for ad in AD) 
	denom = sum(sparsevec(ad.jid,ad.s) for ad in AD)
	grp_q = INDMAT*denom
	grp_qx = INDMAT*num	
	return grp_qx ./ grp_q	
end

function spgetGroupP(AD::Vector{clogit_case_output}, INDMAT::SparseMatrixCSC)
	num = sum(sparsevec(ad.jid, ad.s .* ad.p) for ad in AD) 
	denom = sum(sparsevec(ad.jid,ad.s) for ad in AD)
	grp_q = INDMAT*denom
	grp_rev = INDMAT*num	
	return grp_rev ./ grp_q	
end

spgetGroupQty(AD::Vector{clogit_case_output}, INDMAT::SparseMatrixCSC, M::Real=1) = INDMAT*spgetQty(AD, M)

spgetGroupShares(AD::Vector{clogit_case_output}, INDMAT::SparseMatrixCSC) = INDMAT*spgetShares(AD)

function spgetGroupdQdP(AD::Vector{clogit_case_output}, INDMAT::SparseMatrixCSC, PdivY::Bool=false)  
	(G,J) = size(INDMAT)
	dQdP = spgetdQdP(AD, PdivY)
	dQjdPj = diag(dQdP)
	grp_dQdP = spzeros(G, G)
	@inbounds for a in 1:G
		grp_dQdP[a,a] = sum(dQjdPj .* INDMAT[a, :])
		for b in 1:G
			if a!==b
				grp_dQdP[a,b] = sum(dQdP.*(INDMAT[a,:]*INDMAT[b,:]'))
			end
		end 
	end
	return grp_dQdP
end 

function spgetGroupDiversionRatioMatrix(AD::Vector{clogit_case_output}, INDMAT::SparseMatrixCSC, PdivY::Bool=false)  
	(G,J) = size(INDMAT)
	dQdP = spgetdQdP(AD, PdivY)
	dQjdPj = diag(dQdP)
	DR = zeros(G, G)
	@inbounds for a in 1:G
		dQAdPA = sum(dQjdPj .* INDMAT[a, :])
		for b in 1:G
			if a!==b
				dQBdPA = sum(dQdP.*(INDMAT[a,:]*INDMAT[b,:]'))
				DR[a,b] = - dQBdPA / dQAdPA 
			end
		end 
	end
	return DR
end 

function spgetGroupElasticityMatrix(AD::Vector{clogit_case_output}, INDMAT::SparseMatrixCSC, PdivY::Bool=false)
	if PdivY 
		Q = spgetGroupQty(AD, INDMAT)
		P = spgetGroupP(AD, INDMAT)	
		dQdP = spgetGroupdQdP(AD, INDMAT, true)
		I = findnz(dQdP)
		return dQdP .* P ./ Q' 
	else 
		Q = spgetGroupQty(AD, INDMAT)
		X = spgetGroupX(AD, INDMAT)	
		dQdX = spgetGroupdQdP(AD, INDMAT, false)
		I = findnz(dQdX)
		return dQdX .* X ./ Q' 
	end
end
