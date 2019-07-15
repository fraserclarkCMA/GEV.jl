
function logsumexp(V::Vector{T}) where T<:Real
	maxV = maximum(V)
	return maxV + log(sum(exp.(V .- maxV)))
end 

std_err(f::Function, x::Vector) = sqrt.(LinearAlgebra.diag(inv(ForwardDiff.hessian(f, x))))

