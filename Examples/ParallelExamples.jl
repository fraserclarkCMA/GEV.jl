# ****** Do a parallel version of clogit example ************* #

using Distributed
addprocs(2)
@everywhere push!(LOAD_PATH, "./Git")
@everywhere using GEV

using CSV, DataFrames, StatsModels, Optim, LinearAlgebra
df = CSV.read("./Examples/Data/restaurant.csv");

# clogit formula
f1 = @formula( chosen ~ cost + distance + rating);

# clogit - Model
clm = clogit_model(f1, df ; case=:family_id, choice_id=:restaurant)

# Conditional Logit
cl = clogit( clm, make_clogit_data(clm, df));

# Option 1. Estimate clogit model with LBFGS() or other algorithm only requiring gradients
result = estimate_clogit(cl; opt_mode = :shared_parallel, 	# <- Need to call :parallel here
							 opt_method = :grad,  	# <- :grad or :hess , linked to algorithm
							x_initial = randn(cl.model.nx),
							algorithm = LBFGS(), 	# <- algorithm
							optim_opts = Optim.Options(show_trace=true), # <- optim options
							batch_size = 50, 		# <- Specify batch size per parallel iteration
							workers = workers());   # <- Can put subset of workers i.e. [2]

# Can also distribute (additional overhead - maybe useful very many choice occasions, many workers and limited memory) 
result = estimate_clogit(cl; opt_mode = :dist_parallel,
							 opt_method = :grad,  
							x_initial = randn(cl.model.nx),
							algorithm = LBFGS(),
							optim_opts = Optim.Options(show_trace=true),
							batch_size = 50,
							workers=[2,3]);

# Option 2. Estimate clogit model with Newton() or other method requiring Hessian
result = estimate_clogit(cl; opt_mode = :parallel,
							 opt_method = :hess,  
							x_initial = randn(cl.model.nx),
							algorithm = Newton(),
							batch_size = 50, 
							optim_opts = Optim.Options(show_trace=true));


# Optimal parameter value
LLstar = -Optim.minimum(result);
xstar = Optim.minimizer(result);
se = sqrt.(diag(inv(pmap_hessian_clogit(xstar, cl.data))));
coeftable = vcat(["Variable" "Coef." "std err"],[clm.coefnames xstar se])

# Print out results - this is working and checks out versus stata!
println("Log-likelihood = $(round(LLstar,digits=4))")
vcat(["Variable" "Coef." "std err"],[cl.model.coefnames xstar se])

#=

 "Variable"    "Coef."    "std err"
 "cost"      -0.154309   0.0173858 
 "distance"  -0.0853461  0.0437514 
 "rating"     0.866979   0.0981221 

=#

