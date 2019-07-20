This is a package estimates Generalized Extreme Value (GEV) models. The project is in the early phases of development and is not documented - amongst other things. 

Currently, the package implement conditional and 2-level nested logit models on datasets supplied by the user in a DataFrame, makes use of StatsModels.jl formula to specify the model and implements using the Optim.jl framework.

Estimation can be implmented in parallel or serial through wrappers for estimation: `estimate_clogit` and `estimate_nlogit`. By default gradients are provided, though the user can choose whether or not to use the Hessian through the `opt_method` option. 

The user can pass Optim.jl options to control which algorithm is used and the `Optim.options()` can also be passed to control other aspects of the optimization.

Below is an example that uses the stata example dataset

'''
using GEV, CSV, DataFrames, Optim, StatsModels

# Load in the data
df = CSV.read("./Git/GEV/Examples/Data/restaurant.csv");

# clogit formula
f1 = @formula( chosen ~ cost + distance + rating);

# clogit - Model
clm = clogit_model(f1, df ; case=:family_id, choice_id=:restaurant)

# Conditional Logit
cl = clogit( clm, make_clogit_data(clm, df));

# Option 1. Estimate clogit model in serial using Gradient
result = estimate_clogit(cl; opt_mode = :serial,
                             opt_method = :grad,  
                            x_initial = randn(cl.model.nx),
                            algorithm = LBFGS(),
                            optim_opts = Optim.Options());

# Option 2. Optimise clogit model in serial using Hessian
result = estimate_clogit(cl; opt_mode = :serial,
                             opt_method = :hess,  
                            x_initial = randn(cl.model.nx),
                            algorithm = Newton(),
                            optim_opts = Optim.Options());

# Result parameter values & se calculations
LLstar = -Optim.minimum(result);
xstar = Optim.minimizer(result);
se = std_err(x->ll_clogit(x,cl), Optim.minimizer(result))
coeftable = vcat(["Variable" "Coef." "std err"],[clm.coefnames xstar se])

# Print out results
println("Log-likelihood = $(round(LLstar,digits=4))")
vcat(["Variable" "Coef." "std err"],[cl.model.coefnames xstar se])
'''

At this early stage other usage examples are stored in the Examples folder. They replicate example from analogous help files in Stata and show how to implement a parallel version (in case the data is very large).

The nested logit is non-convex and so multiple starting points are advisable.