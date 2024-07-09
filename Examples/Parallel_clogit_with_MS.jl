# EXAMPLE -- clogit demand estimation with merger simulation

using Distributed

@everywhere begin
	using Pkg
	Pkg.activate("./Git/GEV.jl")
	Pkg.instantiate()
end

@everywhere using GEV
using CSV, DataFrames, StatsModels, Optim, LinearAlgebra, Statistics

df = CSV.read(joinpath(@__DIR__,"./Git/GEV.jl/Examples/Data/restaurant.csv"), DataFrame);

prods = unique(df.restaurant);
J = length(prods);
prod_df = DataFrame(:restaurant => prods, :pid => Int.(collect(1:length(prods))), :owner=>[1,1,2,2,3,3,4]);
df = leftjoin(df, prod_df, on=:restaurant);
df.cost0 = deepcopy(df.cost);
df.invY = 1 ./ df.income;

# Setup

# clogit formula
f1 = @formula( chosen ~ cost + cost&invY + distance + rating);

# clogit - Model
clm = clogit_model(f1, df ; case=:family_id, choice_id=:pid)

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

# ----------- POST-ESTIMATION DEMAND SIDE OUTPUTS ------------- #

# AGGREGATE DEMAND DERIVATIVE MATRIX AND PURCHASE PROBABILITIES 

df0 = deepcopy(df);
pos_cost = 1
pos_cost_div_Y = 4
AD = AggregateDemand(xstar, df0, cl, pos_cost, pos_cost_div_Y )
AD.PROB
AD.DQ

# AGGREGATE DEMAND DERIVATIVE MATRIX AND PURCHASE PROBABILITIES 
# Evaluate at New Point
PRICE = combine(groupby(df, :restaurant), :cost => mean => :price).price
AD = AggregateDemand(xstar, df0, cl.model, PRICE, :cost, pos_cost, pos_cost_div_Y )
AD.PROB
AD.DQ
CW0 = AD.CW

# FIRM LEVEL DIVERSION RATIO
firm_df = combine(groupby(df0, :pid), :restaurant => unique => :brand, :owner => unique => :owner, :cost => mean => :price)
OWN = make_ownership_matrix(firm_df, :owner)
DR = AggregateDiversionRatioMatrix( AD.DQ , OWN.IND)

# FIRM LEVEL PRICE ELASTICITY MATRIX 
E = AggregateElasticityMatrix(AD.DQ, AD.PROB, PRICE, OWN.IND)

# ----------- SUPPLY SIDE ----------- #

# BACK OUT MARGINAL COSTS
MC = getMC(AD.PROB, PRICE, OWN.MAT, AD.DQ)

# AGGREGATE MULTI-PRODUCT MARGINS (%)
MARGIN = getMARGIN(AD.PROB, PRICE, OWN.IND, OWN.MAT, AD.DQ)

# ------------------ #
# MERGER SIMULATION
# ------------------ #

using NLsolve

# Merger of 3 & 4
firm_df.post_owner = firm_df.owner 
firm_df.post_owner[firm_df.owner.==3] .= 4
POST_OWN= make_ownership_matrix(firm_df, :post_owner)

# FOC under new merger under static Bertrand-nash competition
df1 = deepcopy(df0)

foc(x) = FOC(zeros(J), xstar, df1, cl.model, MC, POST_OWN.MAT, x, :cost, pos_cost, pos_cost_div_Y)

# Solve for post-merger prices (start from pre-merger)
post_res = nlsolve(foc, PRICE)

# Price Rise 
PRICE1 = post_res.zero
PriceIncrease = (PRICE1 .- PRICE ) ./ PRICE

# Consumer Welfare Change
CW0 = AggregateDemand(xstar, df0, cl.model, PRICE, :cost, pos_cost, pos_cost_div_Y ).CW
CW1 = AggregateDemand(xstar, df1, cl.model, PRICE1, :cost, pos_cost, pos_cost_div_Y ).CW
CW_CHANGE = CW1/CW0 - 1

