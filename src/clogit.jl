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

struct passdata{T<:Real}
	theta :: Vector{T}
	id 	  :: Int64
end

function ll_clogit(beta::Vector{T}, cld::clogit_data) where T<:Real
	LL = 0.
	for case_data in cld 
		@unpack jstar, dstar, Xj = case_data
		V = Xj*beta
		LL += V[jstar] - logsumexp(V)
	end
	return -LL
end

ll_clogit(beta::Vector{T}, cl::clogit) where T<:Real = ll_clogit(beta, cl.data)

# For parallel version



function ll_clogit_case(beta::Vector{T}, cld::clogit_data, id::Int64) where T<:Real
	@unpack jstar, dstar, Xj = cld[id]
	V = Xj*beta
	LL = V[jstar] - logsumexp(V)
	return -LL 
end

function ll_clogit_case(beta::Vector{T}, clcd::clogit_case_data) where T<:Real
	@unpack jstar, dstar, Xj = clcd
	V = Xj*beta
	LL = V[jstar] - logsumexp(V)
	return -LL 
end

struct clogit_case{T<:Real} 
	F :: T
	G :: Vector{T}
	H :: Matrix{T}
end

function grad_clogit_case(beta::Vector{T}, cld::clogit_data, id::Int64; useAD::Bool=false) where T<:Real
	ForwardDiff.gradient(x->ll_clogit_case(x, cld, id), beta)
end

function grad_clogit_case(beta::Vector{T}, clcd::clogit_case_data; useAD::Bool=false) where T<:Real
	ForwardDiff.gradient(x->ll_clogit_case(x, clcd), beta)
end

function clogit_prob(beta::Vector{T}, clcd::clogit_case_data) where T<:Real
	@unpack jstar, dstar, Xj = clcd
	V = Xj*beta  
	maxV = maximum(V)
	return  exp.(V .- maxV) ./sum(exp.(V .- maxV))
end

function analytic_grad_clogit_case(beta::Vector{T}, cld::clogit_data, id::Int64) where T<:Real
	@unpack jstar, dstar, Xj = cld[id]
	J,K = size(Xj)
	prob = clogit_prob(beta, cl.data[id])
	grad = view(Xj,jstar,:) .- sum(view(Xj,j,:).*prob[j] for j in 1:J)
	return -grad 
end

function analytic_grad_clogit_case(beta::Vector{T}, clcd::clogit_case_data) where T<:Real
	@unpack jstar, dstar, Xj = clcd
	J,K = size(Xj)
	prob = clogit_prob(beta, clcd)
	grad = view(Xj,jstar,:) .- sum(view(Xj,j,:).*prob[j] for j in 1:J)
	return -grad 
end

function hessian_clogit_case(beta::Vector{T}, cld::clogit_data, id::Int64) where T<:Real
	ForwardDiff.hessian(x->ll_clogit_case(x, cld, id), beta)
end

function hessian_clogit_case(beta::Vector{T}, clcd::clogit_case_data) where T<:Real
	ForwardDiff.hessian(x->ll_clogit_case(x, clcd), beta)
end

function fg_clogit_case(beta::Vector{T}, cld::clogit_data, id::Int64) where T<:Real
	clogit_case(ll_clogit_case(beta, cld, id), grad_clogit_case(beta, cld, id), Matrix{T}(undef,0,0)) 
end

function fg_clogit_case(beta::Vector{T}, clcd::clogit_case_data) where T<:Real
	clogit_case(ll_clogit_case(beta, clcd), grad_clogit_case(beta, clcd), Matrix{T}(undef,0,0)) 
end

function analytic_fg_clogit_case(beta::Vector{T}, cld::clogit_data, id::Int64) where T<:Real
	@unpack jstar, dstar, Xj = cld[id]
	J,K = size(Xj)
	V = Xj*beta
	LL = V[jstar] - logsumexp(V)
	prob = clogit_prob(beta, cld[id])
	grad = view(Xj,jstar,:) .- sum(view(Xj,j,:).*prob[j] for j in 1:J)
	return clogit_case(-LL, -grad, Matrix{Float64}(undef,0,0)) 
end

function analytic_fg_clogit_case(beta::Vector{T}, clcd::clogit_case_data) where T<:Real
	@unpack jstar, dstar, Xj = clcd
	J,K = size(Xj)
	V = Xj*beta
	LL = V[jstar] - logsumexp(V)
	prob = clogit_prob(beta, clcd)
	grad = view(Xj,jstar,:) .- sum(view(Xj,j,:).*prob[j] for j in 1:J)
	return clogit_case(-LL, -grad, Matrix{Float64}(undef,0,0)) 
end

function fgh_clogit_case(beta::Vector{T}, cld::clogit_data, id::Int64) where T<:Real
	clogit_case(ll_clogit_case(beta, cld, id), grad_clogit_case(beta, cld, id), hessian_clogit_case(beta, cld, id)) 
end

function fgh_clogit_case(beta::Vector{T}, clcd::clogit_case_data, id::Int64) where T<:Real
	clogit_case(ll_clogit_case(beta, clcd), grad_clogit_case(beta, clcd), hessian_clogit_case(beta, clcd)) 
end


# Wrapper for Estimation
function estimate_clogit(cl::clogit; opt_mode = :serial, opt_method = :none, grad_type = :analytic,
						x_initial = randn(cl.model.nx), algorithm = LBFGS(), 
						optim_opts = Optim.Options(), workers=workers())

	clos_ll_clogit_case(pd) = ll_clogit_case(pd.theta, cl.data, pd.id)
	clos_grad_clogit_case(pd) = grad_clogit_case(pd.theta, cl.data, pd.id)
	clos_analytic_grad_clogit_case(pd) = analytic_grad_clogit_case(pd.theta, cl.data, pd.id)
	clos_hessian_clogit_case(pd) = hessian_clogit_case(pd.theta, cl.data, pd.id)
	clos_fg_clogit_case(pd) = fg_clogit_case(pd.theta, cl.data, pd.id)
	clos_analytic_fg_clogit_case(pd) = analytic_fg_clogit_case(pd.theta, cl.data, pd.id)
	clos_fgh_clogit_case(pd) = fgh_clogit_case(pd.theta, cl.data, pd.id)

	if opt_mode == :parallel 
		pool = CachingPool(workers)
	
		function pmap_clogit_ll(beta::Vector{T}) where T<:Real 
			PD = [passdata(beta, i) for i in 1:length(cl.data)]	
			sum(pmap(clos_ll_clogit_case, pool, PD))
		end

		function pmap_clogit_analytic_grad(beta::Vector{T}) where T<:Real
			PD = [passdata(beta, i) for i in 1:length(cl.data)]	
			sum(pmap(clos_analytic_grad_clogit_case, pool, PD))
		end

		function pmap_clogit_analytic_fg!(F, G, theta::Vector{T}) where T<:Real
			PD = [passdata(theta, i) for i in 1:length(cl.data)]	
			clc = pmap(clos_analytic_fg_clogit_case, pool, PD)
			if G != nothing
				G[:] = sum([y.G for y in clc])
			end
			if F != nothing
				return sum([y.F for y in clc])
			end
		end

		function pmap_clogit_grad(beta::Vector{T}) where T<:Real
			PD = [passdata(beta, i) for i in 1:length(cl.data)]	
			sum(pmap(clos_grad_clogit_case, pool, PD))
		end

		function pmap_clogit_Hess(beta::Vector{T}) where T<:Real
			PD = [passdata(beta, i) for i in 1:length(cl.data)]	
			sum(pmap(clos_hessian_clogit_case, pool, PD))
		end

		function pmap_clogit_fg!(F, G, beta::Vector{T}) where T<:Real
			PD = [passdata(beta, i) for i in 1:length(cl.data)]	
			clc = pmap(clos_fg_clogit_case, pool, PD)
			if G != nothing
				G[:] = sum([y.G for y in clc])
			end
			if F != nothing
				return sum([y.F for y in clc])
			end
		end

		function pmap_clogit_fgh!(F, G, H, beta::Vector{T}) where T<:Real
			PD = [passdata(beta, i) for i in 1:length(cl.data)]	
			clc = pmap(clos_fgh_clogit_case, pool, PD)
			if H != nothing
				H[:] = sum([y.H for y in clc])
			end
			if G != nothing
				G[:] = sum([y.G for y in clc])
			end
			if F != nothing
				return sum([y.F for y in clc])
			end
		end

		if opt_method == :grad
			if grad_type == :analytic
				out = Optim.optimize(Optim.only_fg!(pmap_clogit_analytic_fg!), x_initial, algorithm, optim_opts)
			else 
				out = Optim.optimize(Optim.only_fg!(pmap_clogit_fg!), x_initial, algorithm, optim_opts)
			end
		elseif opt_method == :hess
			out = Optim.optimize(Optim.only_fgh!(pmap_clogit_fgh!), x_initial, algorithm, optim_opts)
		else 
			out = Optim.optimize(pmap_clogit_ll, x_initial, NelderMead(), optim_opts)
		end

	elseif opt_mode == :serial

		function map_clogit_ll(beta::Vector{T}) where T<:Real 
			PD = [passdata(beta, i) for i in 1:length(cl.data)]	
			sum(map(clos_ll_clogit_case, PD))
		end

		function map_clogit_analytic_grad(beta::Vector{T}) where T<:Real
			PD = [passdata(beta, i) for i in 1:length(cl.data)]	
			sum(map(clos_analytic_grad_clogit_case, PD))
		end

		function map_clogit_analytic_fg!(F, G, theta::Vector{T}) where T<:Real
			PD = [passdata(theta, i) for i in 1:length(cl.data)]	
			clc = map(clos_analytic_fg_clogit_case, PD)
			if G != nothing
				G[:] = sum([y.G for y in clc])
			end
			if F != nothing
				return sum([y.F for y in clc])
			end
		end

		function map_clogit_grad(beta::Vector{T}) where T<:Real
			PD = [passdata(beta, i) for i in 1:length(cl.data)]	
			sum(map(clos_grad_clogit_case, PD))
		end

		function map_clogit_Hess(beta::Vector{T}) where T<:Real
			PD = [passdata(beta, i) for i in 1:length(cl.data)]	
			sum(map(clos_hessian_clogit_case, PD))
		end

		function map_clogit_fg!(F, G, beta::Vector{T}) where T<:Real
			PD = [passdata(beta, i) for i in 1:length(cl.data)]	
			clc = map(clos_fg_clogit_case, PD)
			if G != nothing
				G[:] = sum([y.G for y in clc])
			end
			if F != nothing
				return sum([y.F for y in clc])
			end
		end

		function map_clogit_fgh!(F, G, H, beta::Vector{T}) where T<:Real
			PD = [passdata(beta, i) for i in 1:length(cl.data)]	
			YY = map(clos_fgh_clogit_case, PD)
			if H != nothing
				H[:] = sum([yy.H for yy in YY])
			end
			if G != nothing
				G[:] = sum([yy.G for yy in YY])
			end
			if F != nothing
				return sum([yy.F for yy in YY])
			end
		end

		if opt_method == :grad
			if grad_type == :analytic
				out = Optim.optimize(Optim.only_fg!(map_clogit_analytic_fg!), x_initial, algorithm, optim_opts)
			else 
				out = Optim.optimize(Optim.only_fg!(map_clogit_fg!), x_initial, algorithm, optim_opts)
			end		elseif opt_method == :hess
			out = Optim.optimize(Optim.only_fgh!(map_clogit_fgh!), x_initial, algorithm, optim_opts)
		else 
			out = Optim.optimize(map_clogit_ll, x_initial, NelderMead(), optim_opts)
		end
	else 
		out = Optim.optimize(x->ll_clogit(x, cl.data), x_initial, algorithm, optim_opts; autodiff=:forward )
	end

	return out
end
