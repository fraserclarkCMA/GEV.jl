
#=
	User responsibility to check meet specifications
=#

# ******************************************************** #

using GEV

# Additional packages for examples 
using CSV, DataFrames, Optim, StatsModels

# cd into GEV.jl
; #= path to GEV.jl =#
df = CSV.read("./Examples/Data/restaurant.csv");

# ******************************************************** #

# ************ ESTIMATE CONDITIONAL LOGIT ****************** #
# clogit formula
f1 = @formula( chosen ~ cost + distance + rating);

# clogit - Model
clm = clogit_model(f1, df ; case=:family_id, choice_id=:restaurant)

# Conditional Logit
cl = clogit( clm, make_clogit_data(clm, df));

# Test Code
x0 = rand(cl.model.nx)
clogit_loglike(x0, cl)

# Optimise clogit model
result = optimize(x->clogit_loglike(x,cl), zeros(clm.nx); autodiff = :forward)

# Optimal parameter value
LLstar = -Optim.minimum(result);
xstar = Optim.minimizer(result);
se = std_err(x->clogit_loglike(x,cl), Optim.minimizer(result))
coeftable = vcat(["Variable" "Coef." "std err"],[clm.coefnames xstar se])

# Print out results
println("Log-likelihood = $(round(LLstar,digits=4))")
vcat(["Variable" "Coef." "std err"],[cl.model.coefnames xstar se])
