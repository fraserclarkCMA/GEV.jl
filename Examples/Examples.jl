#=# Dev code for GEV
; cd Git/GEV 
] activate .
		
using DataFrames, DataFramesMeta, CSV, StatsModels, 
		Optim, ForwardDiff, Parameters
using LinearAlgebra
VV{T} = Vector{Vector{T}}
include("./src/clogit.jl")
include("./src/nlogit.jl")
include("./src/utils.jl")
=#

# cd to Examples/Data/
using GEV

#=
	User responsibility to check meet specifications
=#

# ******************************************************** #

using GEV

using CSV, DataFrames, Optim, StatsModels
df = CSV.read("./Git/GEV/Examples/Data/restaurant.csv");

# clogit formula
f1 = @formula( chosen ~ cost + distance + rating);

# clogit - Model
clm = clogit_model(f1, df ; case=:family_id, choice_id=:restaurant)

# Conditional Logit
cl = clogit( clm, make_clogit_data(clm, df));

# Two options - the first one default to use autodiff

# Option 1. Estimate clogit model 
result = estimate_clogit(cl; opt_mode = :serial,
							 opt_method = :grad,  
							x_initial = randn(cl.model.nx),
							algorithm = LBFGS(),
							optim_opts = Optim.Options());

# Option 2. Optimise clogit model directly through Optim
result = optimize(x->clogit_loglike(x,cl), zeros(clm.nx); autodiff = :forward)

# Optimal parameter value
LLstar = -Optim.minimum(result);
xstar = Optim.minimizer(result);
se = std_err(x->ll_clogit(x,cl), Optim.minimizer(result))
coeftable = vcat(["Variable" "Coef." "std err"],[clm.coefnames xstar se])

# Print out results - this is working and checks out versus stata!
println("Log-likelihood = $(round(LLstar,digits=4))")
vcat(["Variable" "Coef." "std err"],[cl.model.coefnames xstar se])

# ********************* ADD NESTS TO THE DATAFRAME ************************* #

nestlist = ["fast","fast","family","family","family","fancy","fancy"];
nestframe = DataFrame(:nestid=>levels(nestlist), :nestnum=>collect(1:length(levels(nestlist));));
jlist =  ["Freebirds","MamasPizza", "CafeEccell","LosNortenos","WingsNmore","Christophers","MadCows"];
nests = DataFrame(:restaurant=>jlist,:nestid=>nestlist);
nests = join(nests,nestframe, on=:nestid);
categorical!(nests, :nestid);
df = join(df,nests,on=:restaurant);

# Information needed in estimation
number_of_nests = length(levels(nestlist));
nest_labels = sort!(unique(df[:,[:nestnum, :nestid]]), :nestnum)[:nestid]

# ********************** ESTIMATE: NL MODEL 1 ******************************* #

# Nested Logit - Model 1
f1 = @formula( chosen ~ cost + distance + rating);
nlm1 = nlogit_model(f1, df; case=:family_id, nests = :nestnum, choice_id=:family_id, 
							RUM=false, num_lambda=number_of_nests, nest_labels=nest_labels); 

# Nested Logit - Model 1
nl1 = nlogit(nlm1, make_nlogit_data(nlm1, df));

# Optimize
opt1 = optimize(x->ll_nlogit(x, nl1), rand(nl1.model.nx), BFGS(); autodiff=:forward)
xstar1 = Optim.minimizer(opt1);
se1 = std_err(x->ll_nlogit(x, nl1), xstar1  );
LL1 = ll_nlogit(xstar1, nl1);

println("Log-likelihood = $(round(LL1,digits=4))")
vcat(["Variable" "Coef." "std err"],[nl1.model.coefnames xstar1 se1])


# *************************  ESTIMATE: NL MODEL 2 **************************** #

# Nested Logit - Model 2
f2 = @formula( chosen ~ income&nestid);
nlm2 = nlogit_model(f1, f2, df; case=:family_id, nests = :nestnum, choice_id=:family_id, 
									RUM=false, num_lambda=number_of_nests, nest_labels=nest_labels); 

# Nested Logit
nl2 = nlogit(nlm2, make_nlogit_data(nlm2, df));

# Optimize
opt2 = optimize(x->ll_nlogit(x, nl2), rand(nl2.model.nx), BFGS(); autodiff=:forward)
xstar2 = Optim.minimizer(opt2);
se2 = std_err(x->ll_nlogit(x, nl2), xstar2  );
LL2 = ll_nlogit(xstar2, nl2);

println("Log-likelihood = $(round(LL2,digits=4))")
vcat(["Variable" "Coef." "std err"],[nl2.model.coefnames xstar2 se2])


# ****************************** ESTIMATE: NL MODEL 3 *********************************** #

# Nested Logit - Model 3
f1 = @formula( chosen ~ cost + distance + rating);
f3 = @formula( chosen ~ income&nestid + kids&nestid);
nlm3 = nlogit_model(f1, f3, df; 
					case=:family_id, nests = :nestnum, choice_id=:family_id, 
					RUM=false, num_lambda=number_of_nests, nest_labels=nest_labels); 

# Nested Logit - with data
nl3 = nlogit(nlm3, make_nlogit_data(nlm3, df));

# Optimize
opt3 = optimize(x->ll_nlogit(x, nl3), rand(nl3.model.nx), BFGS(); autodiff=:forward)
xstar3 = Optim.minimizer(opt3);
se3 = std_err(x->ll_nlogit(x, nl3), xstar3  );
LL3 = ll_nlogit(xstar3, nl3);

println("Log-likelihood = $(round(LL3,digits=4))")
vcat(["Variable" "Coef." "std err"],[nl3.model.coefnames xstar3 se3])

# ADD A MULTISTART!
