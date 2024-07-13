function make_ownership_matrix(df::DataFrame, groupvar::Symbol)

	indmat = StatsBase.indicatormat(df[!, groupvar])
	return ( IND = indmat, MAT = indmat'*indmat)

end

function FOC(F, beta::Vector, df::DataFrame, clm::clogit_model, MC::Vector, OMEGA::Matrix{Int64},
				P, xvarname::Symbol, xvarpos::Int64, interacted_xvarpos::ScalarOrVector{Int64}=Int64[])

	AD = AggregateDemand(beta, df, clm, P, xvarname, xvarpos, interacted_xvarpos)

	F = AD.PROB .+ (OMEGA.*AD.DQ)*(P - MC)

end

# New function call to allow for interacted price and individual characteristic in merger sim
function FOC(F, beta::Vector, df::DataFrame, clm::clogit_model, MC::Vector, OMEGA::Matrix{Int64},
				P, individual_varname::Symbol, xvarname::Symbol, 
				xvarpos::Int64, interacted_xvarpos::ScalarOrVector{Int64}=Int64[])

	AD = AggregateDemand(beta, df, clm, P, individual_varname, xvarname, xvarpos, interacted_xvarpos)

	F = AD.PROB .+ (OMEGA.*AD.DQ)*(P - MC)

end

getMC(prob::Vector, price::Vector, OWN::Matrix, DQ::Matrix) = price + (OWN .* DQ)\prob

function getMARGIN(prob::Vector, price::Vector, IND::Matrix, OWN::Matrix, DQ::Matrix)
	FIRM_SHARES = IND .* prob'
	PROFIT = -FIRM_SHARES*((OWN.*DQ)\prob) 
	REVENUE =  FIRM_SHARES*price
	return PROFIT ./ REVENUE;
end 

