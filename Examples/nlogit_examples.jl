
#=
	User responsibility to check meet specifications
=#

# ******************************************************** #

using GEV

# Additional packages for examples 
using CSV, DataFrames, Optim, StatsModels

# cd into GEV.jl
 #= path to GEV.jl =#
; cd Git/GEV
df = CSV.read("./Examples/Data/restaurant.csv");

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
Random.seed!(12345)
opt1 = estimate_nlogit(nl1 ; opt_mode = :serial,
							 opt_method = :grad,  
							x_initial = randn(nl1.model.nx),
							algorithm = LBFGS(),
							optim_opts = Optim.Options());

# Output
xstar1 = Optim.minimizer(opt1);
se1 = std_err(x->ll_nlogit(x, nl1), xstar1  );
LL1 = -Optim.minimum(opt1);

println("Log-likelihood = $(round(LL1,digits=4))")
vcat(["Variable" "Coef." "std err"],[nl1.model.coefnames xstar1 se1])


# *************************  ESTIMATE: NL MODEL 2 **************************** #

# Nested Logit - Model 2 - use Newton + Hessian for this and price optimisation trace
f2 = @formula( chosen ~ income&nestid);
nlm2 = nlogit_model(f1, f2, df; case=:family_id, nests = :nestnum, choice_id=:family_id, 
									RUM=false, num_lambda=number_of_nests, nest_labels=nest_labels); 

# Nested Logit
nl2 = nlogit(nlm2, make_nlogit_data(nlm2, df));

# Optimize
Random.seed!(12345)
opt2 = estimate_nlogit(nl2 ; opt_mode = :serial,
							 opt_method = :grad,  
							x_initial = rand(nl2.model.nx),
							algorithm = LBFGS(),
							optim_opts = Optim.Options(show_trace=true))

# Output
xstar2 = Optim.minimizer(opt2);
se2 = std_err(x->ll_nlogit(x, nl2), xstar2  );
LL2 = -Optim.minimum(opt2)

println("Log-likelihood = $(round(LL2,digits=4))")
vcat(["Variable" "Coef." "std err"],[nl2.model.coefnames xstar2 se2])


# ****************************** ESTIMATE: NL MODEL 3 *********************************** #

# Nested Logit - Model 3
f1 = @formula( chosen ~ cost + distance + rating);
f3 = @formula( chosen ~ income&nestid + kids&nestid);
nlm3 = nlogit_model(f1, f3, df; case=:family_id, nests = :nestnum, choice_id=:family_id, 
					RUM=false, num_lambda=number_of_nests, nest_labels=nest_labels); 

# Nested Logit - with data
nl3 = nlogit(nlm3, make_nlogit_data(nlm3, df));

# Optimize
Random.seed!(12345)
opt3 = estimate_nlogit(nl3 ; opt_mode = :serial,
							 opt_method = :grad,  
							x_initial = rand(nl3.model.nx),
							algorithm = LBFGS(),
							optim_opts = Optim.Options());
# Output
xstar3 = Optim.minimizer(opt3);
se3 = std_err(x->ll_nlogit(x, nl3), xstar3  );
LL3 = -Optim.minimum(opt3);

println("Log-likelihood = $(round(LL3,digits=4))")
vcat(["Variable" "Coef." "std err"],[nl3.model.coefnames xstar3 se3])

# Can also implement a multi-start to guard against non-convexity of nlogit

# Define function for any start value
estnl(x0) = estimate_nlogit(nl3 ; opt_mode = :serial,
							 opt_method = :grad,  
							x_initial = x0,
							algorithm = LBFGS(),
							optim_opts = Optim.Options())
# Multistart version
NumStarts = 10;
multiopt = map(estnl, [randn(nl3.model.nx) for i in 1:NumStarts]);
opt3 = multiopt[findmin(map(Optim.minimum, multiopt))[2]]

xstar3 = Optim.minimizer(opt3);
se3 = std_err(x->ll_nlogit(x, nl3), xstar3  );
LL3 = -Optim.minimum(opt3);

println("Log-likelihood = $(round(LL3,digits=4))")
vcat(["Variable" "Coef." "std err"],[nl3.model.coefnames xstar3 se3])

