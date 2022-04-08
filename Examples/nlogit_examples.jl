
#=
	User responsibility to check meet specifications
=#

# ******************************************************** #

using Pkg
Pkg.activate("./Git/GEV")
Pkg.instantiate()

using GEV

# Additional packages for examples 
using CSV, DataFrames, Optim, StatsModels, Random, StatsBase, CategoricalArrays

# cd into GEV.jl
#= ; cd path to GEV.jl =#
df = CSV.read("./Git/GEV/Examples/Data/restaurant.csv", DataFrame);

# ********************* ADD NESTS TO THE DATAFRAME ************************* #

nestlist = ["fast","fast","family","family","family","fancy","fancy"];
nestframe = DataFrame(:nestid=>levels(nestlist), :nestnum=>collect(1:length(levels(nestlist));));
jlist =  ["Freebirds","MamasPizza", "CafeEccell","LosNortenos","WingsNmore","Christophers","MadCows"];
nests = DataFrame(:restaurant=>jlist,:nestid=>nestlist);
nests = innerjoin(nests,nestframe, on=:nestid);
nests.nestid = categorical(nests.nestid);
df = innerjoin(df,nests,on=:restaurant);

# Information needed in estimation
number_of_nests = length(levels(nestlist));
nest_labels = sort!(unique(df[:,[:nestnum, :nestid]]), :nestnum).nestid

# ********************** ESTIMATE: NL MODEL 1 ******************************* #

#=

# Equivalent Stata Code #

webuse restaurant
nlogitgen type = restaurant(fast: Freebirds | MamasPizza, family:  CafeEccell | LosNortenos | WingsNmore, fancy : Christophers | MadCows)
nlogit chosen cost distance rating || type: , base(family) || restaurant:, noconst case(family_id)

=#

# Nested Logit - Model 1
f1 = @formula( chosen ~ cost + distance + rating);
nlm1 = nlogit_model(f1, df; case=:family_id, nests = :nestnum, choice_id=:restaurant, 
							RUM=false, num_lambda=number_of_nests, nest_labels=nest_labels); 

# Nested Logit - Model 1
nl1 = nlogit(nlm1, make_nlogit_data(nlm1, df));

# Optimize
Random.seed!(12345)
opt1 = estimate_nlogit(nl1 ; opt_mode = :serial,
							 opt_method = :grad,
							 grad_type = :analytic,  
							x_initial = rand(nl1.model.nx),
							algorithm = BFGS(),
							optim_opts = Optim.Options());

# Output
xstar1 = Optim.minimizer(opt1);
se1 = std_err(x->ll_nlogit(x, nl1), xstar1  );
LL1 = -Optim.minimum(opt1);

println("Log-likelihood = $(round(LL1,digits=4))")
vcat(["Variable" "Coef." "std err"],[nl1.model.coefnames xstar1 se1])

# ********* Post-estimation *********** #

# Purchase Probabilities
nl1.model.opts[:outside_share] = 0.5
prob_df = nlogit_prob(xstar1, nl1);
prob_df = innerjoin(prob_df, nestframe, on=nl1.model.nest_id )
df = leftjoin(df, prob_df, on=[nl1.model.case_id, nl1.model.nest_id, nl1.model.choice_id]; makeunique=true);

# Check RUM 
prob_df[!, :TS1] = TS1.(prob_df[!, :pr_g]);
prob_df[!, :TS2] = TS2.(prob_df[!, :pr_g]);
RUMtest = combine(groupby(prob_df,[:nestid, nl1.model.nest_id]), 
	:TS1 => minimum => :TS1, :TS2 => minimum => :TS2)
RUMtest[!, :lambda] = nl1.model.params.lambda[RUMtest[!, nl1.model.nest_id]];
RUMtest[!, :isRUM] = RUMtest.lambda .< RUMtest.TS2;
RUMtest

# Elasticities for cost (note RUM 0<λg<1 doesn't hold... but check formats)
price_indices = [1]
ejj = elas_own_nlogit(xstar1, nl1, price_indices);
ekjg = elas_within_nlogit(xstar1, nl1, price_indices);
ekj = elas_across_nlogit(xstar1, nl1, price_indices);

elasdf = ejj;
elasdf = innerjoin(elasdf, 
			ekjg[!, [nl1.model.case_id, nl1.model.nest_id, nl1.model.choice_id, :ekjg ]], 
				on=[nl1.model.case_id, nl1.model.nest_id, nl1.model.choice_id]);
elasdf = innerjoin(elasdf, 
			ekj[!, [nl1.model.case_id, nl1.model.nest_id, nl1.model.choice_id, :ekj ]], 
			on=[nl1.model.case_id, nl1.model.nest_id, nl1.model.choice_id]);
elasdf

# Unweighted gradients of elasticities
#∇e_jj = grad_elas_own_nlogit(xstar1, nl1, price_indices);
#∇e_kjg = grad_elas_within_nlogit(xstar1, nl1, price_indices);
#∇e_kj = grad_elas_across_nlogit(xstar1, nl1, price_indices);

# Note: (note RUM 0<λg<1 doesn't hold... but check formats)

# Suppose I have some weights 
elasdf[!, :sample_wgts] = 100*rand(nrow(df));

# Calculate elasticities: unweighted and weighted by choice option
combine(groupby(elasdf,:restaurant)) do x
	DataFrame(
		:ejj => mean(x.ejj), 
		:wgt_ejj => mean(x.ejj, weights(x.sample_wgts)),
  		:ekjg=>mean(x.ekjg), 
  		:wgt_ekjg=>mean(x.ekjg, weights(x.sample_wgts)),
  		:ekj=>mean(x.ekj), 
  		:wgt_ekj=>mean(x.ekj, weights(x.sample_wgts)) 
	)
end


# *************************  ESTIMATE: NL MODEL 2 **************************** #

#=

# Equivalent Stata Code #

webuse restaurant
nlogitgen type = restaurant(fast: Freebirds | MamasPizza, family:  CafeEccell | LosNortenos | WingsNmore, fancy : Christophers | MadCows)
nlogit chosen cost distance rating || type: income , base(family) || restaurant:, noconst case(family_id)

=#

# Nested Logit - Model 2 - use Newton + Hessian for this and price optimisation trace
f2 = @formula( chosen ~ income&nestid);
nlm2 = nlogit_model(f1, f2, df; case=:family_id, nests = :nestnum, choice_id=:restaurant, 
									RUM=false, num_lambda=number_of_nests, nest_labels=nest_labels); 

# Nested Logit
nl2 = nlogit(nlm2, make_nlogit_data(nlm2, df));

# Optimize
Random.seed!(12345)
opt2 = estimate_nlogit(nl2 ; opt_mode = :serial,
							 opt_method = :grad,  
							 grad_type = :analytic,  
							x_initial = rand(nl2.model.nx),
							algorithm = BFGS(),
							optim_opts = Optim.Options(show_trace=true))

# Output
xstar2 = Optim.minimizer(opt2);
se2 = std_err(x->ll_nlogit(x, nl2), xstar2  );
LL2 = -Optim.minimum(opt2)

println("Log-likelihood = $(round(LL2,digits=4))")
vcat(["Variable" "Coef." "std err"],[nl2.model.coefnames xstar2 se2])


# ****************************** ESTIMATE: NL MODEL 3 *********************************** #

#=

# Equivalent Stata Code #

webuse restaurant
nlogitgen type = restaurant(fast: Freebirds | MamasPizza, family:  CafeEccell | LosNortenos | WingsNmore, fancy : Christophers | MadCows)
nlogit chosen cost distance rating || type: income kids, base(family) || restaurant:, noconst case(family_id)

Note: Optim.jl finds marginally lower log-likelihood hence small differences in some coefficients. 

=#

# Nested Logit - Model 3
f1 = @formula( chosen ~ cost + distance + rating);
f3 = @formula( chosen ~ income&nestid + kids&nestid);
nlm3 = nlogit_model(f1, f3, df; case=:family_id, nests = :nestnum, choice_id=:restaurant, 
					RUM=false, num_lambda=number_of_nests, nest_labels=nest_labels); 

# Nested Logit - with data
nl3 = nlogit(nlm3, make_nlogit_data(nlm3, df));

# Optimize
Random.seed!(12345)
opt3 = estimate_nlogit(nl3 ; opt_mode = :serial,
							 opt_method = :grad,  
							 grad_type = :analytic,  
							x_initial = rand(nl3.model.nx),
							algorithm = BFGS(),
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
