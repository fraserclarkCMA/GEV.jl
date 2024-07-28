# EXAMPLE -- clogit demand estimation with merger simulation

using Distributed

addprocs(3)

@everywhere begin
	using Pkg
	Pkg.activate("./Git/GEV.jl")
	Pkg.instantiate()
end

@everywhere using GEV
using CSV, DataFrames, StatsModels, Optim, LinearAlgebra, Statistics , SparseArrays

df = CSV.read(joinpath(@__DIR__,"./Git/GEV.jl/Examples/Data/restaurant.csv"), DataFrame);

df[!, :include] .= 1
for row in eachrow(df)
	if row.chosen .== 0 && rand() > 0.85
		row.include = 0 
	end 
end

df0 = unique(select(df, :family_id, :income, :kids));
df0[!, :restaurant] .= "zzzzzz";
df0[!, :cost] .= 0;
df0[!, :chosen] .= 0;
df0[!, :rating] .= 0;
df0[!, :distance] .= 0;
df0[!, :include] .= 1;

append!(df, df0)
sort!(df, [:family_id, :restaurant])

df = @subset(df, :include .== 1)

prods = sort!(unique(df.restaurant));
J = length(prods);
prod_df = DataFrame(:restaurant => prods, :pid => Int(3) .*Int.(collect(1:length(prods))), :owner=>[1,1,2,2,3,3,4,5]);
df = leftjoin(df, prod_df, on=:restaurant);

# This gives interaction, which then enters as an interaction 
#df.cost0 = deepcopy(df.cost);
df.invY = 1 ./ df.income;
df.cost_div_Y = df.cost .* df.invY;

# ----- #
# Setup #
# ----- #

# clogit formula
f1 = @formula( chosen ~ cost_div_Y + distance + rating);

# clogit - Model
clm = clogit_model(f1, df ; case=:family_id, choice_id=:pid)
clm.opts[:PdivY] = true
clm.opts[:pvar] = :cost
clm.opts[:zvar] = :invY

# Conditional Logit
cl = clogit( clm, make_clogit_data(clm, df));

# Option 1. Estimate clogit model with LBFGS() or other algorithm only requiring gradients
result = estimate_clogit(cl; opt_mode = :parallel, 	# <- Need to call :parallel here
							 opt_method = :grad,  	# <- :grad or :hess , linked to algorithm
							x_initial = randn(cl.model.nx),
							algorithm = GEV.Optim.LBFGS(), 	# <- algorithm
							optim_opts = GEV.Optim.Options(show_trace=true), # <- optim options
							batch_size = 50);   # <- Can put subset of workers i.e. [2]

# Optimal parameter value
LLstar = -Optim.minimum(result);
xstar = Optim.minimizer(result);
se = std_err(xstar, cl)
#se = sqrt.(diag(inv(pmap_hessian_clogit(xstar, cl.data))));
coeftable = vcat(["Variable" "Coef." "std err"],[clm.coefnames xstar se])

# Print out results - this is working and checks out versus stata!
println("Log-likelihood = $(round(LLstar,digits=4))")
vcat(["Variable" "Coef." "std err"],[cl.model.coefnames xstar se])

# ----------- POST-ESTIMATION DEMAND SIDE OUTPUTS AT A NEW PRICE POINT ------------- #

# Make record of DataFrame
df0 = deepcopy(df); 

# No interactions with individual characteristics
pos_price = 1 # Must be an Int()

# Aggregate Demand from raw data
AD = AggregateDemand(xstar, df0, cl, pos_price);

# Number of Products
J = maxJ(AD);

# Prices								
P0 = spgetX(AD, J);

# clogit with new price to test code
cl0 = new_clogit_data(df0, clm, Vector(P0), :cost); 

# Aggregate at new price
AD0 = AggregateDemand(xstar, df0, cl0, pos_price); 	

# ----------------------------------------------------------------------------------- #

# Product level outputs in sparse form
# -------------------------------------

Q = spgetQty(AD0, J)
s = spgetShares(AD0, J)
P = spgetX(AD0, J)
dQdP = spgetdQdP(AD0, J)
dsdP = dQdP ./ length(AD)
DR = spgetDiversionRatioMatrix(AD0, J)
E = spgetElasticityMatrix(AD0, J)

# ----------- POST-ESTIMATION GROUPED DEMAND SIDE OUTPUTS ------------- #


# Grouped level outputs - INCL OUTSIDE GOOD
# --------------------------------------------

# Get group product selector matrix - INCLUDING OUTSIDE GOOD AS J+1 PRODUCT ID AND OUTSIDE GOOD AS N+1 OWNER ID
firm_df = combine(groupby(df0, :pid), :restaurant => unique => :brand, :owner => unique => :owner)
OWN = make_ownership_matrix(firm_df, :owner, :pid)

# FIRM LEVEL
P_g = spgetGroupX(AD0, J, OWN.IND)
Q_g = spgetGroupQty(AD0, J, OWN.IND)
s_g = spgetGroupShares(AD0, J, OWN.IND)
dQdP_g = spgetGroupdQdP(AD0, J, OWN.IND)
dsdP_g = dQdP ./ length(AD0)
DR_g = spgetGroupDiversionRatioMatrix( AD0 , J, OWN.IND)
E_g = spgetGroupElasticityMatrix(AD0, J, OWN.IND)

# Grouped level outputs - ONLY INSIDE GOODS
# -------------------------------------

# Call Inside Goods as dense Vector, Matrices with inside_good_idx as pid ref 
inside_good_idx = getInsideGoods(AD0, J) 

Q_ig = spgetQty(AD0, J, inside_good_idx)
s_ig = spgetShares(AD0, J, inside_good_idx)
P_ig = spgetX(AD0, J, inside_good_idx)
dQdP_ig = spgetdQdP(AD0, J, inside_good_idx)
dsdP_ig = dQdP_ig ./ length(AD)
DR_ig = spgetDiversionRatioMatrix(AD0, J, inside_good_idx)
E_ig = spgetElasticityMatrix(AD0, J, inside_good_idx)

# Putting OG subs on main diagonal
DR_ig .+ I(length(inside_good_idx)).*( 1 .- sum(DR_ig, dims=2))

# ----------- POST-ESTIMATION SUPPLY-SIDE OUTPUTS ------------- #

# Get Marginal Costs
# ------------------------------------------- 

# Option 1: call only inside goods
INDMAT_ig = OWN.IND[1:end-1, inside_good_idx]
OMEGA_ig = OWN.MAT[inside_good_idx, inside_good_idx]
MC = getMC(P_ig, Q_ig, dQdP_ig, Matrix(OMEGA_ig))

# Option 2: Call sparse inputs with index with inside good product id's
MC = spgetMC(P,Q,dQdP,OWN.MAT,inside_good_idx) # MC = P .+ (OWN.MAT .* dQdP) \ Q

# Margins under different market structure
# ------------------------------------------- 

NumProds = length(inside_good_idx)
MARGIN_SPN = getMARGIN(P_ig, Q_ig, dQdP_ig, Matrix(I(NumProds)), Matrix(I(NumProds)))
[MARGIN_SPN -1 ./ diag(E_ig) isapprox.(MARGIN_SPN .- -1 ./ diag(E_ig), 0; atol=1e-6)] # Check

MARGIN_MPN = getMARGIN(P_ig, Q_ig, dQdP_ig, Matrix(I(NumProds)), Matrix(OMEGA_ig))
[MARGIN_MPN  (P_ig .-MC)./P_ig isapprox.(MARGIN_MPN .- (P_ig .-MC)./P_ig, 0.; atol=1e-6)] # Check

FIRM_MARGIN = getMARGIN(P_ig, Q_ig, dQdP_ig, Matrix(INDMAT_ig), Matrix(OMEGA_ig))

# Now with Sparse
OG_id = maximum(df.pid)

MARGIN_SPN = spgetMARGIN(P, Q, dQdP, sparse(I(OG_id)), sparse(I(OG_id)), inside_good_idx)[inside_good_idx]
[MARGIN_SPN -1 ./ diag(E_ig) isapprox.(MARGIN_SPN .- -1 ./ diag(E_ig), 0; atol=1e-6)] # Check

MARGIN_MPN = spgetMARGIN(P, Q, dQdP, sparse(I(OG_id)), OWN.MAT, inside_good_idx)[inside_good_idx]
[MARGIN_MPN  (P_ig .-MC)./P_ig isapprox.(MARGIN_MPN .- (P_ig .-MC)./P_ig, 0.; atol=1e-6)] # Check

# Regular calls
FIRM_MARGIN = spgetMARGIN(P, Q, dQdP, OWN.IND, OWN.MAT, inside_good_idx)



