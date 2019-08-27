
function logsumexp(V::Vector{T}) where T<:Real
	maxV = maximum(V)
	return maxV + log(sum(exp.(V .- maxV)))
end 

function multinomial(V::Vector{T}) where T<:Real
	maxV = maximum(V)
	return  exp.(V .- maxV) ./sum(exp.(V .- maxV))
end

std_err(f::Function, x::Vector) = sqrt.(LinearAlgebra.diag(inv(ForwardDiff.hessian(f, x))))

function std_err(beta::Vector{T}, cl::clogit) where T<:Real
	J = length(beta)
	OG = zeros(J,J)
	for cld in cl.data 
		g = analytic_grad_clogit_case(beta, cld) 
		OG .+= g*g'
	end
	if isposdef(OG)
		u,s,v = svd(OG)
		se = sqrt.(LinearAlgebra.diag(v*Diagonal(1.0 ./s)*u'))
	else 
		sv = svdvals(OG)
		nsv = J - sum(isapprox.(sv./sv[1],0.; atol= 1e-6))
		u,s,v = tsvd(OG, nsv)
		se = sqrt.(abs.(LinearAlgebra.diag(v*Diagonal(1.0 ./s)*u')))
		println("WARNING: used tsvd to approx se with nsv=$(nsv)")
	end
	return se
end

function std_err(theta::Vector{T}, nl::nlogit) where T<:Real
	J = length(theta)
	OG = zeros(J,J)
	for id in 1:length(nl.data)
		g = analytic_grad_nlogit_case(theta, nl, id) 
		OG .+= g*g'
	end
	if isposdef(OG)
		u,s,v = svd(OG)
		se = sqrt.(LinearAlgebra.diag(v*Diagonal(1.0 ./s)*u'))
	else 
		sv = svdvals(OG)
		nsv = J - sum(isapprox.(sv./sv[1],0.; atol= 1e-6))
		u,s,v = tsvd(OG, nsv)
		se = sqrt.(abs.(LinearAlgebra.diag(v*Diagonal(1.0 ./s)*u')))
		println("WARNING: used tsvd to approx se with nsv=$(nsv)")
	end
	return se
end
