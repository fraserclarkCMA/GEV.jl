
#=
	User responsibility to check meet specifications
=#

# ******************************************************** #

using GEV

# Additional packages for examples 
using CSV, DataFrames, Optim, StatsModels

# cd into GEV.jl
; cd Git/GEV
#= ; cd path to GEV.jl =#
df = CSV.read("./Examples/Data/restaurant.csv");

# ******************************************************** #

# ************ ESTIMATE CONDITIONAL LOGIT ****************** #
# clogit formula
f1 = @formula( chosen ~ cost + distance + rating);

# clogit - Model
clm = clogit_model(f1, df ; case=:family_id, choice_id=:restaurant)

# Conditional Logit
cl = clogit( clm, make_clogit_data(clm, df));

# Optimise clogit model
result = estimate_clogit(cl ; opt_mode = :serial,
							 opt_method = :grad,
							 grad_type = :analytic,  
							x_initial = randn(cl.model.nx),
							algorithm = LBFGS(),
							optim_opts = Optim.Options());

# Optimal parameter value
LLstar = -Optim.minimum(result);
xstar = Optim.minimizer(result);
se = std_err(x->ll_clogit(x,cl), Optim.minimizer(result));
coeftable = vcat(["Variable" "Coef." "std err"],[clm.coefnames xstar se]);

# Print out results
println("Log-likelihood = $(round(LLstar,digits=4))")
vcat(["Variable" "Coef." "std err"], [cl.model.coefnames xstar se])

# Calculate predicted purchase probabilities
share_outside_good = 0.9;
df[:s_j] = clogit_prob(xstar, cl.data);
df[:s_j_unc] = (1.0 .- share_outside_good).*df[:s_j];

# Calculate elasticities: own, cross
price_vars = [1];
df[:e_jj] = elas_own_clogit(xstar, cl.data, price_vars);
df[:e_kj] = elas_cross_clogit(xstar, cl.data, price_vars);

∇e_jj = grad_elas_own_clogit(xstar, cl.data, price_vars);
∇e_kj = grad_elas_cross_clogit(xstar, cl.data, price_vars);

