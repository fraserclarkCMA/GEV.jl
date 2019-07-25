
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

function ll_clogit_case(beta::Vector{T}, clcd::clogit_case_data) where T<:Real
	@unpack jstar, dstar, Xj = clcd
	V = Xj*beta
	LL = V[jstar] - logsumexp(V)
	return -LL 
end

ll_clogit_case(beta::Vector{T}, cld::clogit_data, id::Int64) where T<:Real = ll_clogit_case(beta, cld[id]) 

function grad_clogit_case(beta::Vector{T}, clcd::clogit_case_data; useAD::Bool=false) where T<:Real
	ForwardDiff.gradient(x->ll_clogit_case(x, clcd), beta)
end

grad_clogit_case(beta::Vector{T}, cld::clogit_data, id::Int64) where T<:Real = grad_clogit_case(beta, cld[id]) 

function clogit_prob(beta::Vector{T}, clcd::clogit_case_data) where T<:Real
	@unpack jstar, dstar, Xj = clcd
	V = Xj*beta  
	maxV = maximum(V)
	return  exp.(V .- maxV) ./sum(exp.(V .- maxV))
end

function analytic_grad_clogit_case(beta::Vector{T}, clcd::clogit_case_data) where T<:Real
	@unpack jstar, dstar, Xj = clcd
	J,K = size(Xj)
	prob = clogit_prob(beta, clcd)
	grad = view(Xj,jstar,:) .- sum(view(Xj,j,:).*prob[j] for j in 1:J)
	return -grad 
end

analytic_grad_clogit_case(beta::Vector{T}, cld::clogit_data, id::Int64) where T<:Real = analytic_grad_clogit_case(beta, cld[id]) 

function fg_clogit_case(beta::Vector{T}, clcd::clogit_case_data) where T<:Real
	clogit_case(ll_clogit_case(beta, clcd), grad_clogit_case(beta, clcd), Matrix{T}(undef,0,0)) 
end

fg_clogit_case(beta::Vector{T}, cld::clogit_data, id::Int64) where T<:Real = fg_clogit_case(beta, cld[id]) 

function analytic_fg_clogit_case(beta::Vector{T}, clcd::clogit_case_data) where T<:Real
	@unpack jstar, dstar, Xj = clcd
	J,K = size(Xj)
	V = Xj*beta
	LL = V[jstar] - logsumexp(V)
	prob = clogit_prob(beta, clcd)
	grad = view(Xj,jstar,:) .- sum(view(Xj,j,:).*prob[j] for j in 1:J)
	return clogit_case(-LL, -grad, Matrix{Float64}(undef,0,0)) 
end

analytic_fg_clogit_case(beta::Vector{T}, cld::clogit_data, id::Int64) where T<:Real = analytic_fg_clogit_case(beta, cld[id]) 

function hessian_clogit_case(beta::Vector{T}, clcd::clogit_case_data) where T<:Real
	ForwardDiff.hessian(x->ll_clogit_case(x, clcd), beta)
end
hessian_clogit_case(beta::Vector{T}, cld::clogit_data, id::Int64) where T<:Real = hessian_clogit_case(beta, cld[id]) 
hessian_clogit_case(dd::distdata) = hessian_clogit_case(dd.theta, dd.data)

function pmap_hessian_clogit(beta::Vector{T}, data::clogit_data; batch_size = 1) where T<:Real
	DD = [distdata(beta, cld) for cld in data]	
	sum(pmap(hessian_clogit_case, DD, batch_size=batch_size))
end

function fgh_clogit_case(beta::Vector{T}, clcd::clogit_case_data) where T<:Real
	clogit_case(ll_clogit_case(beta, clcd), grad_clogit_case(beta, clcd), hessian_clogit_case(beta, clcd)) 
end

fgh_clogit_case(beta::Vector{T}, cld::clogit_data, id::Int64) where T<:Real = fgh_clogit_case(beta, cld[id]) 


