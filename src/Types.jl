# types

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
	opts[:outside_share] = 0.
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

struct clogit 
	model 	:: clogit_model
	data 	:: clogit_data
end

struct passdata{T<:Real}
	theta :: Vector{T}
	id 	  :: Int64
end

struct clogit_case{T<:Real} 
	F :: T
	G :: Vector{T}
	H :: Matrix{T}
end

@with_kw struct nlogit_param
	beta 	:: Vector{Real} 	# Parameters varying within an nest
	alpha 	:: Vector{Real}		# Parameters varying across nests 
	lambda 	:: Vector{Real}		# Nesting Parameters post-transform
end

# Constructor functions for nlogit_param
#nlogit_param() = nlogit_param(Vector{Float64}(),Vector{Float64}(),Vector{Float64}(),Vector{Float64}())
nlogit_param(NX::Int64,NW::Int64,NN::Int64) where T<:Real = nlogit_param(zeros(Float64,NX),zeros(Float64,NW),zeros(Float64,NN))

@with_kw struct nlogit_model
	f_beta 	:: StatsModels.FormulaTerm 	# StatsModels.FormulaTerm for variable that vary within nests 
	f_alpha :: StatsModels.FormulaTerm	# StatsModels.FormulaTerm for variables that vary across nests (f_alpha = @formula(d ~ Z&nestid) or not identifed (i.e. Z might be HH chars))
	params 	:: nlogit_param 	# Parameters
	case_id	:: Symbol 			# 1 if option chosen, 0 if not chosen
	nest_id :: Symbol 			# Identifier for nest identifier variable
	choice_id :: Symbol 		# Choice identifier variable
	idx 	:: Dict 			# Index for parameters
	nx 		:: Int64 			# Number of parameters 
	coefnames :: Vector 		# Coef names
	opts 	:: Dict 			# Group id var, number of different nesting parameters, nest id
	flags	:: Dict 			# Flags indicating which of beta, alpha and tau are estimated	
end

struct nlogit_nest_data
	jstar 		:: Int64 				# Position of chosen option in chosen nest in the data
	dstar 		:: Union{Int64,String}	# Identifier of chosen option 
	nest_star	:: Int64 				# Chosen nest 
	nest_num 	:: Int64
	Xj			:: Matrix{Float64}		# Matrix of regressors for within-nest
	Wk 			:: Matrix{Float64}		# Matrix of regressors for across-nest
end
nlogit_nest_data() = new()

nlogit_case_data = Vector{nlogit_nest_data}
nlogit_data = VV{nlogit_nest_data}

struct nlogit 
	model 	:: nlogit_model
	data 	:: nlogit_data
end

struct nlogit_case{T<:Real} 
	F :: T
	G :: Vector{T}
	H :: Matrix{T}
end

struct distdata{T<:Union{clogit_case_data,nlogit_case_data}}
	theta :: Vector{Float64}
	data  :: T
end
