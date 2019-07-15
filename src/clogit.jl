# Condtional logit model

@with_kw struct clogit_param
	beta 	:: Vector{Real} 	# Parameters varying within an nest
end

# Constructor functions for clogit_param
clogit_param(NX::Int64) where T<:Real = clogit_param(zeros(Float64,NX))

@with_kw struct clogit_model
	f_beta 	:: StatsModels.FormulaTerm 			# StatsModels.Formula for variable that vary within nests 
	params 	:: clogit_param 	# Parameters
	case_id	:: Symbol 			# 1 if option chosen, 0 if not chosen
	choice_id :: Symbol 		# Choice identifier variable
	idx 	:: Dict 			# Index for parameters
	nx 		:: Int64 			# Number of parameters 
	coefnames :: Vector 		# Coefficient names
	opts 	:: Dict 			# Group id var, number of different nesting parameters, nest id
end

# Constructor functions for nlogit_model

function clogit_model(f_beta::StatsModels.FormulaTerm, df::DataFrame; case::Symbol, choice_id::Symbol)
	f_beta = StatsModels.apply_schema(f_beta, StatsModels.schema(f_beta, df))
	NX = size(modelcols(f_beta , df)[2], 2)
	params = clogit_param(NX)
	opts = Dict()
	cfnms = coefnames(f_beta)[2]
	return clogit_model(f_beta, params, case, choice_id, Dict(:beta => 1:NX), NX, cfnms, opts)
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
	f_beta = StatsModels.apply_schema(f_beta, StatsModels.schema(f_beta, df))
	dataset = Vector{clogit_case_data}()
	for casedf in groupby(df, case_id)
		jstar = findall(x->x==1, casedf[f_beta.lhs.sym])[1]
		dstar = casedf[jstar, choice_id]
		Xj = isa(f_beta.rhs.terms[1], StatsModels.InterceptTerm{false}) ?  Matrix{Float64}(undef, 0, 0) : modelcols(f_beta , casedf)[2]
		push!(dataset, clogit_case_data(jstar, dstar, Xj))
	end
	return dataset
end

struct clogit 
	model 	:: clogit_model
	data 	:: clogit_data
end

function clogit_loglike(beta::Vector{T}, cld::clogit_data) where T<:Real
	LL = 0.
	for case_data in cld 
		@unpack jstar, dstar, Xj = case_data
		V = Xj*beta
		LL += V[jstar] - logsumexp(V)
	end
	return -LL
end

clogit_loglike(beta::Vector{T}, cl::clogit) where T<:Real = clogit_loglike(beta, cl.data)
