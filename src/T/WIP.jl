# Internal code

#= 	******************************************************************************

	GEV project overview
	--------------------

	There are n=1,…,N choice in the data. (Assume cross-sectional iid)
	There are i=1,…,J choices in a choice set (Suppress notation to allow choice to differ by group, but it can)

	Response: d_n is a vector with elements ∈ {0,1} and sum(d_n) == 1, j=0 is no purchase option
	Covariates of product j : X_j
	Covariates of n: Z_n (these are observed)
	Parameters: θ = {β, λ} where β are coefficients on covariates, λ are nesting parameters


	In a GEV model:

	Pr(d_n == j) = Y_j*G_j / G

		or 

	ln(Pr(d_n == j) = ln(Y_j) + ln(G_j) - ln(G)

	where Y_j = exp(X_j*β) and G = f(Y) where 

	Comments
	--------
	-> The key here is that f(⋅) varies by GEV model
	-> Main coding idea is to use AD to calculate G_j. 

	****************************************************************************** =#



# Step 1: Create differenced covariates using response as base (Need for AD)

function makeDiffX(f::formula, df::AbstractDataFrame, groupid::Symbol)	
	resp_idx = findall(x -> x == true, df[f.lhs.sym])
	df_resp = df[resp_idx, :]
	noresp_idx = findall(x -> x == false, df[f.lhs.sym])
	X0 = modelmatrix(f, df[noresp_idx, :])
	X1 = [modelmatrix(f, subdf) for subdf in DataFrames.groupby(df_noresp, groupid)]
	ΔX = Matrix{Real}[]
	for (n, id) in enumerate(df_resp[groupid])
		push!(ΔX, X1[n] .- transpose(X0[n, :]))
	end
	return ΔX 
end

# Aliases
VV{T} = Vector{Vector{T}}
VM{T} = Vector{Matrix{T}}

*************************************************************************

abstract type GEVparameters{T<:Real} end

mutable struct clogit_parameters{T} <: GEVparameters{T}
	beta 	:: Vector{T} 	# Non-nest parameters
end

abstract type nlogit_parameters{T} <: GEVparameters{T} end

mutable struct nlogit_parameters{T} <: nlogitparameters{T}
	beta 	:: Vector{T} 	# Non-nest parameters
	mu 		:: T 			# Two-level, Single nest-parameter
end

mutable struct nlogit_parameters{T, R<:Vector{T}} <: nlogitparameters{T}
	beta 	:: Vector{T} 	# Non-nest parameters
	mu 	 	:: R			# Two-levels, nest-specific parameters
end

mutable struct nlogit_parameters{T, R<:VV{T}} <: nlogitparameters{T}
	beta 	:: Vector{T} 	# Non-nest parameters
	mu 	 	:: R 			# More than 2 levels, single nest-parameter
end

# Paired combinatorial logit
mutable struct pcl_parameters{T} <: GEVparameters{T}
	beta 	:: Vector{T} 	# Non-nest parameters
	mu 	 	:: Matrix{T} 	# nest-specific parameters
end

*************************************************************************


mutable struct GEVmodel{T} where T<:Real 
	model 	:: Symbol  			# :clogit, nlogit1con, etc...
	theta 	:: GEVparameters{T} 	# Parameters
	levels 	:: Int64 			# 1 for clogit & PCL, 2+ for nlogit
	nest_membership :: Symbol 	# {:NA, :Full, :Partial} PCL and cross-nested are partial
end

# K number of explanatory vars
function make_clogit(; K::Int64=1, nestparam::Symbol=Symbol("uniform"))
	GEVModel(Symbol("clogit"), clogit_parameters(zeros(K)), 1, Symbol("NA"))
end 

# K number of explanatory vars, L number of nests
function make_nlogit_2L(; K::Int64=1, L::Int64=2, nestparam::Symbol=Symbol("uniform"))
	if nestparam .== :uniform
		GEVModel(Symbol("nlogit"), nlogit_parameters(zeros(K), 0.), Symbol("Full"))
	elseif 
		GEVModel(Symbol("nlogit"), nlogit_parameters(zeros(K), zeros(L)), Symbol("Full"))
	end
end 

#= Add other GEV after using dictioninaries for nest structure =#


# -------------- GEV: Indirect utility function ------------------ #

function funV_clogit(X::Matrix{T}, θ<:GEVparameters{T}) where T<:Real
	@unpack beta = θ 
	return X*beta
end

function funV_GEV(X::Matrix{T}, θ<:GEVparameters{T}) where T<:Real
	@unpack beta, lambda = θ 
	return X*beta
end

# --------------- Y, G, Gi ----------------- #


# 1. Conditional Logit

#=
Step to calculate logit likelihood
----------------------------------

1. Load in dataframe with covariates X, response y, and groupid variable
2. An observative is given by groupid variables -> calculate X*beta for the choice set in groupid
3. Now calculate logsumexp for choice set

=#

# Y function
function funY_clogit(v::Vector{T}) where T<:Real 
	Vmax = maximum(v)
	return exp.(v .- Vmax)
end

# G function
funG_clogit(Y::Vector{T}, θ<:GEVparameters{T}) where T<:Real = log(sum(Y))

# Gi function -> No need to call since log(Gi) = 0 in clogit
funGi_clogit(Y::Vector{T}) where T<:Real = ones(length(Y))

# Loglikelihood for each group
function loglike_it_clogit()
	V - logsumexp
end

# Log-likelihood for the whole sample - already split into groups
function loglike_clogit()
	LL = 0
	for i in #iteratedarrays#
		LL += loglike_it_clogit
	end
	return LL
end

# 2. Nested Logit

Y needs a nestid which i use to select nest parameters

# --------------- conditional logit ----------------- #




function funY_nlogit(v::T, nestid::Int64, θ::GEVparameters{T}, nestid::Int64, nest_membership::Symbol) where T<:Real 
	@unpack beta, lambda = θ
	if nesttype
	Vmax = maximum(v)
	return exp.(v .- Vmax)
end


function funG(Y, θ::GEVparameters, model::Symbol)
	if model .== :clogit
		return sum(Y)
	elseif model .== :nlogit 
		return 
	else 
		throw("You must add GEV model")
	end
end 

function GradientfunG(Y)
return ForwardDiff.gradient(y->funG(y, a), [])










function ... estimate clogit 

	# Input is f, df, groupid
end 

function loglike(beta, ΔX)
	VEctor pf GEV model obs
	ΔXb = [ΔX[n]*beta for n in eachindex(ΔX)]
	Yj = [exp.(ΔXb[n]) for n in eachindex(ΔX)]
	G = [log.(sum(Yj[n])) for n in eachindex(ΔX)]
	Pr = ΔXb - G 

# 1. Estimate clogit model - groupid

abstract type GEVmodel{T<:Real} end
abstract type GEVParameters{T<:Real} end

struct clogit_params{T} <:GEVParameters{T} 
	b :: Vector{T}
end
struct nlogit_params{T} <:GEVParameters{T} 
	b :: Vector{T}
end

struct GEVmodel{T<:Real} 
	theta  :: GEVparameters{T} 
	...
end

struct GEVProbability{T<:Real} 
	Y  :: Vector{T} 
	G  :: T
	Gi :: Vector{T}
end

function clogit(fm::Formula, df::AbstractDataFrame, groupid::Symbol)
    f = apply_schema(f, schema(f, df))
    mf = ModelFrame(fm, df1)
    mm = ModelMatrix(mf)
    mr = model_response(mf)
    #coef = qreg_coef(mr, mm.m, q, s)
    #vcov = qreg_vcov(mr, mm.m, coef, q)
    #stderr = sqrt.(diag(vcov))
    return DataFrameRegressionModel(QRegModel(coef, vcov, stderr, q), mf, mm)
end
qreg(f::Formula, df::AbstractDataFrame, s::Solver) = qreg(f, df, 0.5, s)

coef(x::QRegModel) = x.beta
coef(rm::DataFrameRegressionModel) = coef(rm.model)

vcov(x::QRegModel) = x.vcov

stderr(x::QRegModel) = x.stderr


# Calculate 

categorical!(df1, :y)
fm = @formula y ~ x1 + x2

Input -> dataframe and a formula

Get the Model Matrix -> 

Calculate the model ll(θ) | Data -> Optimisation -> Hessian -> Reporting



So i would need an interface of parameters into b

V_j
Y = exp(V_j) = exp(X*b) = exp(ΔX*b)
G = sum(Y) over j 
G_i = ForwardDiff.gradient(G)

lnPr_j = ln(Y_j) + ln(G_i) - ln(G)
lnPr_j = V_j + ln(G_i) - ln(G)