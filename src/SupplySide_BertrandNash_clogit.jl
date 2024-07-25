

function make_ownership_matrix(df::DataFrame, groupvar::Symbol)
	indmat = StatsBase.indicatormat(df[!, groupvar])
	return ( IND = indmat, MAT = indmat'*indmat)
end

# Make Sparse Ownership Matrix
function make_ownership_matrix(df::DataFrame, groupvar::Symbol, pidvar::Symbol)
	N = maximum(df[!, groupvar])
	J = maximum(df[!, pidvar])
	indmat = spzeros(N,J)
	for row in eachrow(df)
		indmat[row[groupvar], row[pidvar]] = 1
	end
	return ( IND = indmat, MAT = indmat'*indmat)
end

getMC(P::Vector, Q::Vector, dQdP::Matrix, OWN::Matrix) = P + (OWN .* dQdP)\Q

function getMARGIN(Q::Vector, P::Vector, IND::Matrix, OWN::Matrix, dQdP::Matrix)
	FIRM_QTY = IND .* Q'
	PROFIT = -FIRM_QTY*((OWN.*dQdP)\Q) 
	REVENUE =  FIRM_QTY*P
	return PROFIT ./ REVENUE;
end 

function FOC(F, beta::Vector, df::DataFrame, clm::clogit_model, MC::Vector, OMEGA::Matrix{Int64}, P,
				 Pvarname::Symbol, Pvarpos::Int64, parallel::Bool=false)

	# New x -> individual choice sets
	cl = new_clogit_data(df, clm, P, Pvarname)

	# Aggregate Demand
	AD = AggregateDemand(beta, df, cl, Pvarpos, parallel)

	Q = getQty(AD)
	dQdP = getdQdP(AD)

	# FOC
	F = Q .+ (OMEGA.*dQdP)*(P - MC)

end

function FOC(F, beta::Vector, df::DataFrame, clm::clogit_model, MC::Vector, OMEGA::Matrix{Int64}, P,
				 Pvarname::Symbol, Pvarpos::Int64, PZvarpos::Int64, parallel::Bool=false)

	# New x -> individual choice sets
	cl = new_clogit_data(df, clm, P, Pvarname)

	# Aggregate Demand
	AD = AggregateDemand(beta, df, cl, Pvarpos, PZvarpos, parallel)

	Q = getQty(AD)
	dQdP = getdQdP(AD)

	# FOC
	F = Q .+ (OMEGA.*dQdP)*(P - MC)

end

function FOC(F, beta::Vector, df::DataFrame, clm::clogit_model, MC::Vector, OMEGA::Matrix{Int64}, P,
				 Pvarname::Symbol, Pvarpos::Int64, PZvarpos::Vector{Int64}, parallel::Bool=false)

	# New x -> individual choice sets
	cl = new_clogit_data(df, clm, P, Pvarname)

	# Aggregate Demand
	AD = AggregateDemand(beta, df, cl, Pvarpos, PZvarpos, parallel)

	Q = getQty(AD)
	dQdP = getdQdP(AD)

	# FOC
	F = Q .+ (OMEGA.*dQdP)*(P - MC)

end

# Masks to allow for P/Y FOC 
function FOC(F, beta::Vector, df::DataFrame, clm::clogit_model, MC::Vector, OMEGA::Matrix{Int64}, P,
				  xvar::Symbol, pvar::Symbol, zvar::Symbol, xvarpos::ScalarOrVector{Int64}, parallel::Bool=false)

	# New x -> individual choice sets
	cl = new_clogit_data(df, clm, P, xvar, pvar, zvar)

	# Aggregate Demand
	AD = AggregateDemand(beta, df, cl, xvarpos, parallel)

	Q = getQty(AD)
	dQdP = getdQdP(AD, true)

	# FOC
	F = Q .+ (OMEGA.*dQdP)*(P - MC)

end


