# Gradient of nested logit probability

function grad_nlogit_prob(x::Vector{Float64}, θ::nlogit_param, data::nlogit_case_data, flags::Dict, idx::Dict, outside_share::Float64)  
	
	small = eps()
	vec_to_theta!(x, θ, flags, idx)	
	@unpack beta, alpha, lambda = θ

	Nbeta = length(beta)
	Nalpha = length(alpha)
	Numλ = length(lambda)	
	Nx = Nbeta + Nalpha + Numλ

	## Allocate memory
	NumberOfNests = length(data)
	grad_sj = [VV{Float64}() for g in 1:NumberOfNests ]
	grad_sjg = [VV{Float64}() for g in 1:NumberOfNests ]
	grad_sg = [Vector{Float64}() for g in 1:NumberOfNests ]

	s_jg = VV{eltype(x)}()

	D = Vector{Float64}()
	if Nbeta > 1 
		∂D_∂β = VV{Float64}()	
	else  
		∂D_∂β = Vector{Float64}()
	end	
	if flags[:alpha] & Nalpha>1 
		∂D_∂α = VM{Float64}()
	else 
		∂D_∂α = VV{Float64}()
	end

	∂D_∂λ = Vector{Float64}() #= ∂Dg_∂λl = 0 so write Diagonal of Jac Matrix, as vector =#

	nest_idx = Int64[]

	## Compute gradient loop over nest first, then construct across nest components of ll afterwards
	for (nest_ctr, nest_data) in enumerate(data)

		@unpack case_num, jid, jstar, dstar, nest_star, nest_num, Xj, Wk = nest_data

		J,K = size(Xj)
		grad_sjg[nest_ctr] = [zeros(Nx) for j in 1:J]

		if Numλ > 1 
			λ_k = lambda[nest_num] 	# Select the relevant nesting parameter
			∂V_∂λ = zeros(J,Numλ)
		else 
			λ_k = lambda[1]
			∂V_∂λ = zeros(J)
		end
			
		V = Xj*beta /λ_k 	# Jl by 1
		∂V_∂β = Xj ./λ_k 	# J x K of ∂Vj_∂βk = ∂V_∂β[j,k]

		IV = logsumexp(V)  	# Scalar
		push!(s_jg , max.(small, multinomial(V)))  # Jl by 1 -> within group prob s_i|g
		if Nbeta >1
			∂IV_∂β = sum(repeat(s_jg[nest_ctr], 1, K).*∂V_∂β,dims=1)[:]
		else 
			∂IV_∂β = sum(s_jg[nest_ctr].*∂V_∂β,dims=1)[:]
		end

		∂V_∂λ = -V ./λ_k 	# Jl by 1
		∂IV_∂λ = sum(s_jg[nest_ctr].*∂V_∂λ, dims=1)[1]
		push!(nest_idx, nest_num) # Order of nests in the data
		
		if flags[:alpha]
			W = sum(Wk[1,:].*alpha)
			push!(D, W + λ_k*IV)
			push!(∂D_∂α, Wk[1,:])
		else
			push!(D, λ_k*IV)
		end
		push!(∂D_∂β, λ_k*∂IV_∂β)
		push!(∂D_∂λ, IV + λ_k*∂IV_∂λ)

		# Somewhere in here i need a gradient vector for each j in the nest
		@inbounds for j in 1:J
			# Gradient of within group prob wrt :beta
			grad_sjg[nest_ctr][j][idx[:beta]] .+= s_jg[nest_ctr][j].*(∂V_∂β[j,:] .- ∂IV_∂β)

			# Gradient of within group prob wrt :lambda 
			grad_sjg[nest_ctr][j][idx[:lambda][nest_num]] += s_jg[nest_ctr][j].*(∂V_∂λ[j] .- ∂IV_∂λ)
		end
		
	end 

	# Groups: let lnG := logsumexp(D)
	sg = (1.0 .- outside_share).*max.(small, multinomial(D)) # G x 1 vector of nest probs
	∂lnG_∂β = sum((sg.*∂D_∂β)[l]  for l in 1:NumberOfNests)
	if flags[:alpha]
		∂lnG_∂α = sum([sg[l].*∂D_∂α[l] for l in 1:NumberOfNests])
	end	
	∂lnG_∂λ = sg.*∂D_∂λ 

	# Gradients of sg
	for (ctr, nest_num) in enumerate(nest_idx)
		grad_sg[ctr] = zeros(Nx)
		grad_sg[ctr][idx[:beta]] .+= sg[ctr].*(∂D_∂β[ctr] .- ∂lnG_∂β)	
		grad_sg[ctr][idx[:lambda][nest_num]] += sg[ctr]*∂lnG_∂β[nest_num]

		if flags[:alpha]
			grad_sg[ctr][idx[:alpha]] .+= sg[ctr].*(∂D_∂α[ctr] .- ∂lnG_∂α)
		end

		if Numλ>1
			grad_sg[ctr][idx[:lambda][nest_idx]] .-= ∂lnG_∂λ 
		else 
			grad_sg[ctr][idx[:lambda][1]] -= sum(∂lnG_∂λ)
		end

		# Gradient of sj
		grad_sj[ctr] = sg[ctr].*grad_sjg[ctr] .+ s_jg[ctr].*[grad_sg[ctr] for j in 1:length(s_jg[ctr])]
	end 

	# Need to add a method to output in a way linked to the input data 
	out_sj = VV{Float64}()
	for (ctr, ∇sj_nest) in enumerate(grad_sj)
		if ctr == 1
			out_sj = ∇sj_nest
		else 
			append!(out_sj, ∇sj_nest)
		end
	end
	out_sjg = VV{Float64}()
	for (ctr, ∇sjg_nest) in enumerate(grad_sjg)
		if ctr == 1
			out_sjg = ∇sjg_nest
		else 
			append!(out_sjg, ∇sjg_nest)
		end
	end
	out_sg = VV{Float64}()
	for (ctr, ∇sg_nest) in enumerate(grad_sg)
		if ctr == 1
			out_sg = [∇sg_nest for j in 1:size(data[ctr].Xj, 1)]
		else 
			append!(out_sg, [∇sg_nest for j in 1:size(data[ctr].Xj, 1) ])
		end
	end
	return out_sj, out_sjg, out_sg

end

grad_nlogit_prob(x::Vector{T}, model::nlogit_model, data::nlogit_case_data) where T<:Real = 
	grad_nlogit_prob(x, model.params, data, model.flags, model.idx , model.opts[:outside_share])

grad_nlogit_prob(x::Vector{T}, nl::nlogit, case_num::Int64) where T<:Real = grad_nlogit_prob(x, nl.model, nl.data[case_num])

function grad_nlogit_prob(x::Vector{T}, nl::nlogit) where T<:Real
	∇sj = VV{Float64}()
	∇sjg = VV{Float64}()
	∇sg = VV{Float64}()
	for (ctr, case_data) in enumerate(nl.data)
       if ctr==1
       	   (∇sj,∇sjg, ∇sg)= grad_nlogit_prob(x, nl.model, case_data)
       else
			res = grad_nlogit_prob(x, nl.model, case_data)
			append!(∇sj , res[1])
			append!(∇sjg , res[2])
			append!(∇sg , res[3])
       end
    end
	return ∇sj, ∇sjg, ∇sg
end

function fg_nlogit_prob(x::Vector{T}, nl::nlogit) where T<:Real
	∇sj = VV{Float64}()
	∇sjg = VV{Float64}()
	∇sg = VV{Float64}()
	for (ctr, case_data) in enumerate(nl.data)
       if ctr==1
       	   (∇sj,∇sjg, ∇sg)= grad_nlogit_prob(x, nl.model, case_data)
       else
			res = grad_nlogit_prob(x, nl.model, case_data)
			append!(∇sj , res[1])
			append!(∇sjg , res[2])
			append!(∇sg , res[3])
       end
    end
    df = nlogit_prob(x, nl)
    df.grad_sj=∇sj
    df.grad_sjg=∇sjg
    df.grad_sg=∇sg
    return df
end







