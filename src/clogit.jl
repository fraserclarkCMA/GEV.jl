
# Construct the model data set 
function make_clogit_data(model::clogit_model, df::DataFrame)
	@unpack f_beta, params, case_id, choice_id, opts = model 
	f_beta = StatsModels.apply_schema(f_beta, StatsModels.schema(f_beta, df))
	dataset = Vector{clogit_case_data}()	
	for casedf in groupby(df, case_id)
		case_num = casedf[!, case_id][1]
		jid = casedf[!, choice_id]
		jstar = findall(x->x==1, casedf[!, f_beta.lhs.sym])[1]
		dstar = casedf[jstar, choice_id]
		Xj = isa(f_beta.rhs.terms[1], StatsModels.InterceptTerm{false}) ?  Matrix{Float64}(undef, 0, 0) : modelcols(f_beta , casedf)[2]
		push!(dataset, clogit_case_data(case_num , jid, jstar, dstar, Xj))
	end
	return dataset
end

function ll_clogit(beta::Vector{T}, cld::clogit_data) where T<:Real
	LL = 0.
	for case_data in cld 
		@unpack case_num, jid, jstar, dstar, Xj = case_data
		V = Xj*beta
		LL += V[jstar] - logsumexp(V)
	end
	return -LL
end

ll_clogit(beta::Vector{T}, cl::clogit) where T<:Real = ll_clogit(beta, cl.data)

function ll_clogit_case(beta::Vector{T}, clcd::clogit_case_data) where T<:Real
	@unpack case_num, jid, jstar, dstar, Xj = clcd
	V = Xj*beta
	LL = V[jstar] - logsumexp(V)
	return -LL 
end

ll_clogit_case(beta::Vector{T}, cld::clogit_data, id::Int64) where T<:Real = ll_clogit_case(beta, cld[id]) 

function grad_clogit_case(beta::Vector{T}, clcd::clogit_case_data; useAD::Bool=false) where T<:Real
	ForwardDiff.gradient(x->ll_clogit_case(x, clcd), beta)
end

grad_clogit_case(beta::Vector{T}, cld::clogit_data, id::Int64) where T<:Real = grad_clogit_case(beta, cld[id]) 

function analytic_grad_clogit_case(beta::Vector{T}, clcd::clogit_case_data) where T<:Real
	@unpack case_num, jid, jstar, dstar, Xj = clcd
	J,K = size(Xj)
	V = Xj*beta  
	maxV = maximum(V)
	prob = exp.(V .- maxV) ./sum(exp.(V .- maxV))
	grad = view(Xj,jstar,:) .- sum(view(Xj,j,:).*prob[j] for j in 1:J)
	return -grad 
end

analytic_grad_clogit_case(beta::Vector{T}, cld::clogit_data, id::Int64) where T<:Real = analytic_grad_clogit_case(beta, cld[id]) 

function fg_clogit_case(beta::Vector{T}, clcd::clogit_case_data) where T<:Real
	clogit_case(ll_clogit_case(beta, clcd), grad_clogit_case(beta, clcd), Matrix{T}(undef,0,0)) 
end

fg_clogit_case(beta::Vector{T}, cld::clogit_data, id::Int64) where T<:Real = fg_clogit_case(beta, cld[id]) 

function analytic_fg_clogit_case(beta::Vector{T}, clcd::clogit_case_data) where T<:Real
	@unpack case_num, jid, jstar, dstar, Xj = clcd
	J,K = size(Xj)
	V = Xj*beta
	LL = V[jstar] - logsumexp(V)
	V = Xj*beta  
	maxV = maximum(V)
	prob = exp.(V .- maxV) ./sum(exp.(V .- maxV))
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


function clogit_prob(beta::Vector{T}, clcd::clogit_case_data, outside_share::Float64, case_id::Symbol, choice_id::Symbol) where T<:Real
	@unpack case_num, jid, jstar, dstar, Xj = clcd
	J = size(Xj,1)
	V = Xj*beta  
	maxV = maximum(V)
	return  DataFrame(case_id=>case_num*ones(Int64,J), 
						choice_id=>jid,
							:pr_j=>(1.0 .- outside_share).*exp.(V .- maxV) ./sum(exp.(V .- maxV)))
end

function clogit_prob(beta::Vector{T}, cld::clogit_data, outside_share::Float64, case_id::Symbol, choice_id::Symbol) where T<:Real
	DF = DataFrame[]
	for case_data in cld 
		push!(DF, clogit_prob(beta, case_data, outside_share, case_id, choice_id) )
	end
	outdf = deepcopy(DF[1])
	@inbounds for n in 2:length(DF)
		append!(outdf, DF[n])
	end
	return outdf
end

clogit_prob(beta::Vector{T}, cl::clogit) where T<:Real = clogit_prob(beta, cl.data, cl.model.opts[:outside_share], cl.model.case_id, cl.model.choice_id)



#=
function grad_clogit_prob(beta::Vector{T}, clcd::clogit_case_data, outside_share::Float64=0.) where T<:Real
	ForwardDiff.gradient(x->clogit_prob(x, clcd, outside_share), beta)
end

grad_clogit_prob(beta::Vector{T}, cl::clogit) where T<:Real = grad_clogit_prob(beta, cl.data, cl.model.opts[:outside_share])
=#
