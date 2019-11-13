
#=
	User responsibility to check meet specifications
=#

# ******************************************************** #

using Pkg
Pkg.activate("./Git/GEV")
Pkg.instantiate()

using GEV

# Additional packages for examples 
using CSV, DataFrames, Optim, StatsModels

# cd into GEV.jl
df = CSV.read("./Git/GEV/Examples/Data/restaurant.csv");

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
							algorithm = BFGS(),
							optim_opts = Optim.Options());

# Optimal parameter value
LLstar = -Optim.minimum(result);
xstar = Optim.minimizer(result);
se = std_err(xstar,cl);
coeftable = vcat(["Variable" "Coef." "std err"],[clm.coefnames xstar se]);

# Print out results
println("Log-likelihood = $(round(LLstar,digits=4))")
vcat(["Variable" "Coef." "std err"], [cl.model.coefnames xstar se])

# Calculate predicted purchase probabilities
cl.model.opts[:outside_good] = 0.9;
prob_df = clogit_prob(xstar, cl);
df = join(df, prob_df, on=[cl.model.case_id, cl.model.choice_id], kind=:left);

# Calculate elasticities: own, cross
price_vars = [1];
ejj = elas_own_clogit(xstar, cl, price_vars);
ekj = elas_cross_clogit(xstar, cl, price_vars);
elasdf = ejj;
elasdf = join(elasdf, ekj, on=[cl.model.case_id, cl.model.choice_id], kind=:left);
df = join(df, elasdf, on=[cl.model.case_id, cl.model.choice_id], kind=:left);

#=
∇e_jj = grad_elas_own_clogit(xstar, cl, price_vars);
∇e_kj = grad_elas_cross_clogit(xstar, cl, price_vars);
=#
