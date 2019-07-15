
# Functions for clogit.jl

#= Idea is to call 

gdf = groupby(df, :groupid)
for each gdf
	Inputs: DataFrame+Formula->(X, jstar) for group & θ
	Z, Vbar = fZ_clogit(X, θ)
	LL += logProb_clogit(Z, Vbar, jstar)
end 

LL is objective function , θ are parameters

=#
# clogit G function
fG_clogit(Y::Vector{T}) where  T<:Real = sum(Y)

# ************* NOT ROBUST TO OVERFLOW ***************** #

function fY_clogit(X::Matrix{T}, θ::clogit_parameters{T}) where T<:Real
	@unpack beta = θ
	return  exp.(X*beta)
end 

function fY_clogit(V::Vector{T}) where T<:Real
	return  exp.(V)
end 

∇G_clogit(Y::Vector{T}) where T<:Real = ForwardDiff.gradient(fG_clogit, Y)

# clogit (log(∇Gj) == 0 so save calculation and omit)
Prob_clogit(Y::Vector{T}, j::Int64) where T<:Real = Y[j] .* ∇G_clogit(Y)[j] ./ fG_clogit(Y)
logProb_clogit(Y::Vector{T}, j::Int64) where T<:Real = log(Y[j]) - log(fG_clogit(Y))

# ************** ROBUST TO OVERFLOW *******************

# Now with robust calc - want to work with differenced valuations
# Never want to calculate exp(maximum(V))
# Work with z = exp(V - maximum(V))

function fZ_clogit(X::Matrix{T}, θ::clogit_parameters{T}) where T<:Real
	@unpack beta = θ
	V = X*beta	
	maxV = maximum(V)
	return  exp.(V .- maxV), maxV
end 

# Y = exp(maxV).*Z
function fZ_clogit(V::Vector{T}) where T<:Real
	maxV = maximum(V)
	return  exp.(V .- maxV), maxV
end 

function ∇G_clogit(Z::Vector{T}, vmax::T ) where T<:Real
	fGnew(x) = fG_clogit(x, vmax)
	ForwardDiff.gradient(fGnew, Z)
end

Prob_clogit(Z::Vector{T}, maxV::T, j::Int64) where T<:Real = Z[j] .* ∇G_clogit(Z, maxV)[j] ./ fG_clogit(Z, maxV)
logProb_clogit(Z::Vector{T}, maxV::T, j::Int64) = log(Z[j]) - log(fG_clogit(Z, maxV))




