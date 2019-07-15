# Two-level nested logit

#=
User responsibility to check meet specifications - 
=#

@with_kw struct clogit_param
	beta 	:: Vector{Real} 	# Parameters varying within an nest
end

# Constructor functions for clogit_param
clogit_param() = clogit_param(Vector{Float64}())
clogit_param(NX::Int64) where T<:Real = clogit_param(zeros(Float64,NX))

@with_kw struct clogit_model
	f_beta 	:: FormulaTerm 			# Formula for variable that vary within nests 
	params 	:: clogit_param 	# Parameters
	case_id	:: Symbol 			# 1 if option chosen, 0 if not chosen
	choice_id :: Symbol 		# Choice identifier variable
	opts 	:: Dict 			# Group id var, number of different nesting parameters, nest id
end

# Constructor functions for nlogit_model

function clogit_model(f_beta::FormulaTerm, df::DataFrame; case::Symbol, choice_id::Symbol)
	f_beta = apply_schema(f_beta, schema(f_beta, df))
	NX = size(modelcols(f_beta , df)[2], 2)
	params = clogit_param(NX)
	opts = Dict()
	return clogit_model(f_beta, params, case, choice_id, opts)
end

struct clogit_case_data
	jstar 		:: Int64 				# Position of chosen option in chosen nest in the data
	dstar 		:: Union{Int64,String}	# Identifier of chosen option 
	Xj			:: Matrix{Float64}		# Matrix of regressors for within-nest
end
clogit_case_data() = new()

clogit_data = Vector{clogit_case_data}

# Construct the model data set 
function make_clogit_data(model::clogit_model, df::DataFrame)
	@unpack f_beta, params, case_id, choice_id, opts = model 
	f_beta = apply_schema(f_beta, schema(f_beta, df))
	dataset = Vector{clogit_case_data}()
	for casedf in groupby(df, case_id)
		jstar = findall(x->x==1, casedf[f_beta.lhs.sym])[1]
		dstar = casedf[jstar, choice_id]
		Xj = isa(f_beta.rhs.terms[1], StatsModels.InterceptTerm{false}) ?  Matrix{Float64}(undef, 0, 0) : modelcols(f_beta , casedf)[2]
		push!(dataset, clogit_case_data(jstar, dstar, Xj))
	end
	return dataset
end

#=
function logsumexp(V::Vector{T}) where T<:Real
	maxV = maximum(V)
	return maxV + log(sum(exp.(V .- maxV)))
end 
=#

function clogit_loglike(beta::Vector{T}, cld::clogit_data) where T<:Real
	LL = 0.
	for case_data in cld 
		@unpack jstar, dstar, Xj = case_data
		V = Xj*beta
		LL += V[jstar] - logsumexp(V)
	end
	return -LL
end

# Fit nested logit
function fit_clogit(clm::clogit_model, df::DataFrame)
	# Make a data set
	cld = make_clogit_data(clm, df)  

	nx = length(clm.params.beta)
	# Callbacks: objective function
	clogit_loglike(x) = clogit_loglike(x, nld) 

	# Optimize
	return optimize(clogit_loglike, zeros(nx), LBFGS(); autodiff = :forward)
end 

# Get standard errors using ForwardDiff.Hessian 

# Report results using coefnames(f_beta) , coefnames(f_alpha), ["Nest $(i)" for i in levels(nestid)] # check last one order
