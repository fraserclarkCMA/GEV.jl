# ****** Do a parallel version of clogit example ************* #

# julia10 --project=./Git/GEV --depwarn=no

using Distributed
addprocs(2; exeflags="--project=./Git/GEV" )    #= This activates the project environment on startup (i.e. ] -> (GEV) pkg)  prompt)  =#

@everywhere push!(LOAD_PATH, "./Git")
@everywhere using Pkg
@everywhere cd("./Git/GEV")
@everywhere Pkg.activate(".")
@everywhere Pkg.instantiate()
@everywhere using GEV
 

using CSV, DataFrames, StatsModels, Optim, LinearAlgebra
#=
cd(@__DIR__)
df = CSV.read("./Data/restaurant.csv");
=#
df = CSV.read("./Git/GEV/Examples/Data/restaurant.csv");

# clogit formula
f1 = @formula( chosen ~ cost + distance + rating);

# clogit - Model
clm = clogit_model(f1, df ; case=:family_id, choice_id=:restaurant)

# Conditional Logit
cl = clogit( clm, make_clogit_data(clm, df));

# Option 1. Estimate clogit model with LBFGS() or other algorithm only requiring gradients
result = estimate_clogit(cl; opt_mode = :parallel, 	# <- Need to call :parallel here
							 opt_method = :gradientfree,  	# <- :grad or :hess , linked to algorithm
							x_initial = randn(cl.model.nx),
							algorithm = LBFGS(), 	# <- algorithm
							optim_opts = Optim.Options(show_trace=true), # <- optim options
							batch_size = 50);   # <- Can put subset of workers i.e. [2]

# Optimal parameter value
LLstar = -Optim.minimum(result);
xstar = Optim.minimizer(result);
se = sqrt.(diag(inv(pmap_hessian_clogit(xstar, cl.data))));
coeftable = vcat(["Variable" "Coef." "std err"],[clm.coefnames xstar se])


# Option 2. Estimate clogit model with LBFGS() or other algorithm only requiring gradients
result = estimate_clogit(cl; opt_mode = :parallel, 	# <- Need to call :parallel here
							 opt_method = :grad,  	# <- :grad or :hess , linked to algorithm
							x_initial = randn(cl.model.nx),
							algorithm = LBFGS(), 	# <- algorithm
							optim_opts = Optim.Options(show_trace=true), # <- optim options
							batch_size = 50);   # <- Can put subset of workers i.e. [2]

# Optimal parameter value
LLstar = -Optim.minimum(result);
xstar = Optim.minimizer(result);
se = sqrt.(diag(inv(pmap_hessian_clogit(xstar, cl.data))));
coeftable = vcat(["Variable" "Coef." "std err"],[clm.coefnames xstar se])

# Option 3. Estimate clogit model with Newton() or other method requiring Hessian
result = estimate_clogit(cl; opt_mode = :parallel,
							 opt_method = :hess,  
							x_initial = randn(cl.model.nx),
							algorithm = Newton(),
							batch_size = 50, 
							optim_opts = Optim.Options(show_trace=true));

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

