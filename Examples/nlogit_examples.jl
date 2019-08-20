
#=
	User responsibility to check meet specifications
=#

# ******************************************************** #

using GEV

# Additional packages for examples 
using CSV, DataFrames, Optim, StatsModels, Random, StatsBase

# cd into GEV.jl
; cd Git/GEV
#= ; cd path to GEV.jl =#
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
							 grad_type = :analytic,  
							x_initial = randn(nl1.model.nx),
							algorithm = LBFGS(),
							optim_opts = Optim.Options());

# Output
xstar1 = Optim.minimizer(opt1);
se1 = std_err(x->ll_nlogit(x, nl1), xstar1  );
LL1 = -Optim.minimum(opt1);

println("Log-likelihood = $(round(LL1,digits=4))")
vcat(["Variable" "Coef." "std err"],[nl1.model.coefnames xstar1 se1])

# ********* Post-estimation *********** #

# Purchase Probabilities
df[:s_j_cond], df[:s_jg], df[:s_g_cond] = nlogit_prob(xstar1, nl1);

# Allow for outside good share (default is 0%)
nl1.model.opts[:outside_share] = 0.5
df[:s_j_unc], _, df[:s_g_unc] = nlogit_prob(xstar1, nl1);

# Elasticities for cost (note RUM 0<λg<1 doesn't hold... but check formats)
price_indices = [1]
df[:e_own] = elas_own_nlogit(xstar1, nl1, price_indices);
df[:e_within] = elas_within_nlogit(xstar1, nl1, price_indices);
df[:e_across] = elas_across_nlogit(xstar1, nl1, price_indices);

# Unweighted gradients of elasticities
∇e_jj = grad_elas_own_nlogit(xstar1, nl1, price_indices);
∇e_kjg = grad_elas_within_nlogit(xstar1, nl1, price_indices);
∇e_kj = grad_elas_across_nlogit(xstar1, nl1, price_indices);

# Note: (note RUM 0<λg<1 doesn't hold... but check formats)

# Suppose I have some weights 
df[:sample_wgts] = 100*rand(nrow(df));

# Calculate elasticities: unweighted and weighted by choice option
by(df, 
	:restaurant,
	 [:e_own, :sample_wgts] => x->(e_own=mean(x.e_own), wgt_e_own=mean(x.e_own, weights(x.sample_wgts)))
)

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
							 grad_type = :analytic,  
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
							 grad_type = :analytic,  
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
							 grad_type = :analytic,  
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


# Pick one model to do post-estimation


# Calculate predicted purchase probabilities
share_inside_goods = 0.1;
df[:s_j_cond] = clogit_prob(xstar, cl.data);
df[:s_j_unc] = share_inside_goods.*df[:s_j_cond];

# Calculate elasticities: own, cross
df[:e_jj] = elas_own_clogit(xstar, cl.data, 1);
df[:e_kj] = elas_cross_clogit(xstar, cl.data, 1);
