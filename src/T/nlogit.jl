
# Functions for clogit.jl

#= Idea is to call 

# Nest structure given by vector of positions in the data
nests = [ [1,2,3]
		 [4,5,6]
		 [8,9] ];

gdf = groupby(df, :groupid)
for each subDataFrame
	Inputs: DataFrame+Formula->(X, jstar) for group & θ
	Z, Vbar = fZ(X, θ)
	LL += logProb(Z, Vbar, jstar)
end 

LL is objective function , θ are parameters

=#

# Consider logit transform for lambda
fun_sigma(x::Float64) = 1.0 .+ exp.(-x)
fun_lambda(x::Float64) = 1.0 ./ (1.0 .+ exp.(-x))
fun_sigma(x::Vector{Float64}) = 1.0 .+ exp.(-x)
fun_lambda(x::Vector{Float64}) = 1.0 ./ (1.0 .+ exp.(-x))

# nlogit G function
function fG(Y::Vector{T}, θ::nlogit_parameters{T}, nests::VV{Int64}) where  T<:Real
	@unpack beta, mu = θ

	# Transform nest parameters
	lambda = fun_lambda(mu)
	sigma = 1./lambda
	
	# Calculate G for nested logit
	G = 0.
	if length(lambda) > 1   #= Multiple Nest Parameters =#
		for (l,nestidx) in enumerate(nests) 
			G += sum(Y[nestidx].^sigma[l]).^lambda[l]
		end
	else  					#= Common nesting parameter =#
		for nestidx in nests 
			G += sum(Y[nestidx].^sigma).^lambda
		end
	end
	return G
end

# ************* NOT ROBUST TO OVERFLOW ***************** #

function fY(X::Matrix{T}, θ::nlogit_parameters{T}) where T<:Real
	@unpack beta, mu = θ
	return  exp.(X*beta)
end 

function fY(V::Vector{T}) where T<:Real
	return  exp.(V)
end 

function ∇G(Y::Vector{T}, θ::nlogit_parameters{T}, nests::VV{Int64}) where T<:Real
	fGnew(Y) = fG(Y,θ,nests)
	return ForwardDiff.gradient(fGnew, Y)
end

# General Case with j denoted chosen option location in choice set
function Prob(Y::Vector{T}, θ::nlogit_parameters{T}, nests::VV{Int64}, j::Int64) where T<:Real
	return Y[j] * ∇G(Y,θ,nests)[j]) / fG(Y,θ,nests)
end

function logProb(Y::Vector{T}, θ::nlogit_parameters{T}, nests::VV{Int64}, j::Int64) where T<:Real
	return log(Y[j]) + log(∇G(Y,θ,nests)[j]) - log(fG(Y,θ,nests))
end

# ************** ROBUST TO OVERFLOW *******************

# Now with robust calc - want to work with differenced valuations
# Never want to calculate exp(maximum(V))
# Work with z = exp(V - maximum(V))

function fZ(X::Matrix{T}, θ::clogit_parameters{T}) where T<:Real
	@unpack beta = θ
	V = X*beta	
	maxV = maximum(V)
	return  exp.(V .- maxV), maxV
end 

# Y = exp(maxV).*Z
function fZ(V::Vector{T}) where T<:Real
	maxV = maximum(V)
	return  exp.(V .- maxV), maxV
end 

function ∇G(Z::Vector{T}, vmax::T ) where T<:Real
	fGnew(x) = fG(x, vmax)
	ForwardDiff.gradient(fGnew, Z)
end

Prob(Z::Vector{T}, maxV::T, j::Int64) where T<:Real = Z[j] .* ∇G(Z, maxV)[j] ./ fG(Z, maxV)
logProb_clogit(Z::Vector{T}, maxV::T, j::Int64) = log(Z[j]) - log(fG(Z, maxV))

# General Case with j denoted chosen option location in choice set
#logProb(Z::Vector{T}, maxV::T, j::Int64) = log(Z[j]) + log(∇G(Z, maxV)[j]) - log(fG(Z, maxV))



