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
User responsibility to check meet specifications - 
=#

using CSV , DataFrames, Optim, StatsModels

# Inputs: data, [choices,nestid]
df2 = CSV.read("./Git/GEV/Examples/Data/restaurant.csv");

# clogit
f1 = @formula( chosen ~ cost + distance + rating);

clm = clogit_model(f1, df2 ; case=:family_id, choice_id=:restaurant)

cld = make_clogit_data(clm, df2);  

nx = length(clm.params.beta)

# Test Code
x0 = rand(nx)
clogit_loglike(x0, cld)

# Optimise clogit model
result = optimize(x->clogit_loglike(x,cld), zeros(nx); autodiff = :forward)

# Optimal parameter value
LLstar = -Optim.minimum(result)
xstar = Optim.minimizer(result)

stderr = std_err(x->clogit_loglike(x,cld), Optim.minimizer(result))
coeftable = vcat(["Variable" "Coef." "std err"],
	[coefnames(StatsModels.apply_schema(f1, StatsModels.schema(f1,df2)))[2] xstar stderr])

# Print out results - this is working and checks out versus stata!
LLstar
coeftable

# ********************* ADD NESTS TO THE DATAFRAME ************************* #

# Here starts the nlogit estimation code
# Code from here for data (nestframe only necessary if non-ordered Int nest id)

# If multiple nesting parameters
nestlist = ["fast","fast","family","family","family","fancy","fancy"];
nestframe = DataFrame(:nestid=>levels(nestlist), :nestnum=>collect(1:length(levels(nestlist));));
# If a common nesting parameter
# nestframe = DataFrame(:nestid=>levels(nestlist), :nestnum=>ones(length(levels(nestlist))))

jlist =  ["Freebirds","MamasPizza", "CafeEccell","LosNortenos","WingsNmore","Christophers","MadCows"];
nests = DataFrame(:restaurant=>jlist,:nestid=>nestlist);
nests = join(nests,nestframe, on=:nestid);
categorical!(nests, :nestid);
df2 = join(df2,nests,on=:restaurant);

number_of_nests = length(levels(nestlist));

# ********************** ESTIMATE: NL MODEL 1 ******************************* #

nlm1 = nlogit_model(f1, df2; case=:family_id, nests = :nestnum, choice_id=:family_id, 
							RUM=false, num_lambda=number_of_nests); 

# Index linking optimisation and param type
idx1, nx1 = get_vec_dict(nlm1);

# logit data
nld1 = make_nlogit_data(nlm1, df2);  

# Optimize
opt1 = optimize(x->nlogit_loglike(x, nlm1, nld1, idx1), rand(nx1), BFGS(); autodiff=:forward)

xstar1 = Optim.minimizer(opt1)

# *************************  ESTIMATE: NL MODEL 2 **************************** #

f_nm2 = @formula( chosen ~ income&nestid);

nlm2 = nlogit_model(f1, f_nm2, df2; case=:family_id, nests = :nestnum, choice_id=:family_id, 
									RUM=false, num_lambda=number_of_nests); 

# Index linking optimisation and param type
idx2, nx2 = get_vec_dict(nlm2);

# logit data
nld2 = make_nlogit_data(nlm2, df2);  

# Optimize
opt2 = optimize(x->nlogit_loglike(x, nlm2, nld2, idx2), rand(nx2), BFGS(); autodiff=:forward)
xstar2 = Optim.minimizer(opt2)
se2 = std_err(x->nlogit_loglike(x, nlm2, nld2, idx2), xstar2  )
LL2 = nlogit_loglike(xstar2, nlm2, nld2, idx2)

f1s = apply_schema(f1, schema(f1,df2));
f2s = apply_schema(f_nm2, schema(f_nm2,df2))
coefnms = vcat(coefnames(f1s)[2],coefnames(f2s)[2],["tau$(i)" for i in 1:3]) 

[coefnms xstar2 se2]

# ************************************************************************* #
f_nm3 = @formula( chosen ~ income&nestid + kids&nestid);
nlm3 = nlogit_model(f1, f_nm3, df2; case=:family_id, nests = :nestnum, choice_id=:family_id, 
									RUM=false, num_lambda=number_of_nests); 

# Index linking optimisation and param type
idx3, nx3 = get_vec_dict(nlm3);

# logit data
nld3 = make_nlogit_data(nlm3, df2);  

# Optimize
opt3 = optimize(x->nlogit_loglike(x, nlm3, nld3, idx3), rand(nx3), BFGS(); autodiff=:forward)
xstar3 = Optim.minimizer(opt3)
se3 = std_err(x->nlogit_loglike(x, nlm3, nld3, idx3), xstar3  )
LL3 = nlogit_loglike(xstar3, nlm3, nld3, idx3)

f1s = apply_schema(f1, schema(f1,df2));
f3s = apply_schema(f_nm3, schema(f_nm3,df2))
coefnms3 = vcat(coefnames(f1s)[2],coefnames(f3s)[2],["tau$(i)" for i in 1:3]) 

[coefnms3 xstar3 se3]








# Not convex - so do a multistart!
opt = [optimize(nlogit_loglike, rand(nx); autodiff=:forward) for i in 1:100];
LLmulti = map(x->nlogit_loglike(Optim.minimizer(x)),opt)
(LLmin, resnum) = findmin(LLmulti)
result = opt[resnum]
xstar = Optim.minimizer(result)
# LL @ xstar
LLstar = nlogit_loglike(xstar)

# Check gradient vector @ xstar is zero
∇Gstar = ForwardDiff.gradient(nlogit_loglike, xstar)
isapprox.(∇Gstar, 0.; atol=1e-3)
# Get standard Errors
std_err = sqrt.(LinearAlgebra.diag(inv(ForwardDiff.hessian(nlogit_loglike, xstar))))

[xstar std_err]
# ************************* #

nlm.opts[:RUM] = true

x0 = -1000*rand(nx)
nlogit_loglike(x0)
ForwardDiff.gradient(nlogit_loglike, x0)

res = fit_nlogit(nlm, df2);  

