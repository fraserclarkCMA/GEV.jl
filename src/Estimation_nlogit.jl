# Wrapper for Estimation
function estimate_nlogit(nl::nlogit; opt_mode = :serial, opt_method = :none, grad_type = :analytic,
						x_initial = randn(cl.model.nx), algorithm = LBFGS(), batch_size = 1, optim_opts = Optim.Options(), movedata = true)

	if opt_mode == :parallel 

		out = estimate_nlogit_parallel(nl, opt_method, grad_type, x_initial, algorithm, batch_size, optim_opts, movedata)

	else

		out = estimate_nlogit_serial(nl, opt_method, grad_type, x_initial, algorithm , optim_opts)

	end 

	return out 

end 

# Wrappers for Estimation assuming nl -> Main.nl on worker - useful in Parallel 
# Note caching pool unique to pmap call, i want multiple pmap calls so this should be more efficient despite global
ll_nlogit_case(beta::Vector{T}, id::Int64) where T<:Real = ll_nlogit_case(beta, Main.nl.model, Main.nl.data[id])
grad_nlogit_case(beta::Vector{T}, id::Int64) where T<:Real = grad_nlogit_case(beta,Main.nl.model, Main.nl.data[id]) 
analytic_grad_nlogit_case(beta::Vector{T}, id::Int64) where T<:Real = analytic_grad_nlogit_case(beta,Main.nl.model, Main.nl.data[id]) 
hessian_nlogit_case(beta::Vector{T}, id::Int64) where T<:Real = hessian_nlogit_case(beta, Main.nl.model, Main.nl.data[id]) 
fg_nlogit_case(beta::Vector{T}, id::Int64) where T<:Real = fg_nlogit_case(beta, Main.nl.model, Main.nl.data[id]) 
analytic_fg_nlogit_case(beta::Vector{T}, id::Int64) where T<:Real = analytic_fg_nlogit_case(beta, Main.nl.model, Main.nl.data[id]) 
fgh_nlogit_case(beta::Vector{T}, id::Int64) where T<:Real = fgh_nlogit_case(beta, Main.nl.model, Main.nl.data[id]) 

function estimate_nlogit_parallel(nl::nlogit, opt_method::Symbol, grad_type::Symbol, x_initial::Vector{Float64}, 
					algorithm, batch_size::Int64, optim_opts, movedata)

	# Step 1: Copy the data to Main.nl_model Main.nl_data on all workers
	if movedata
		printstyled("\nTransferring data to workers....\n"; bold=true, color=:blue)
		sendto(workers(), nl = nl)
		printstyled("Transfer of data to workers compelete\n"; bold=true, color=:blue)
	else 
		printstyled("Warning: You should manually move data\n"; bold=true, color=:blue)
	end

	# Define vector 
	NUMOBS = length(nl.data)

	# Optimisation functions for master process
	pmap_nlogit_ll(beta::Vector{T}) where T<:Real = 
		sum(pmap(m->ll_nlogit_case(beta, m), 1:NUMOBS, batch_size=batch_size))

	pmap_nlogit_analytic_grad(beta::Vector{T}) where T<:Real =
		sum(pmap(m->analytic_nlogit_case(beta, m), 1:NUMOBS, batch_size=batch_size))

	function pmap_nlogit_analytic_fg!(F, G, beta::Vector{T}) where T<:Real
		clc = pmap(m->analytic_fg_nlogit_case(beta,m), 1:NUMOBS, batch_size=batch_size)
		if G != nothing
			G[:] = sum([y.G for y in clc])
		end
		if F != nothing
			return sum([y.F for y in clc])
		end
	end

	pmap_nlogit_grad(beta::Vector{T}) where T<:Real = 
		sum(pmap(m->grad_nlogit_case(beta,m), 1:NUMOBS, batch_size=batch_size))

	pmap_nlogit_Hess(beta::Vector{T}) where T<:Real =
		sum(pmap(m->hessian_nlogit_case(beta,m), 1:NUMOBS, batch_size=batch_size))

	function pmap_nlogit_fg!(F, G, beta::Vector{T}) where T<:Real
		clc = pmap(m->fg_nlogit_case(beta,m), 1:NUMOBS, batch_size=batch_size)
		if G != nothing
			G[:] = sum([y.G for y in clc])
		end
		if F != nothing
			return sum([y.F for y in clc])
		end
	end

	function pmap_nlogit_fgh!(F, G, H, beta::Vector{T}) where T<:Real
		clc = pmap(m->fgh_nlogit_case(beta, m), 1:NUMOBS, batch_size=batch_size)
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

	# Step 3: Do optimisation 
	if opt_method == :grad
		if grad_type == :analytic
			out = Optim.optimize(Optim.only_fg!(pmap_nlogit_analytic_fg!), x_initial, algorithm, optim_opts)
		else 
			out = Optim.optimize(Optim.only_fg!(pmap_nlogit_fg!), x_initial, algorithm, optim_opts)
		end
	elseif opt_method == :hess
		out = Optim.optimize(Optim.only_fgh!(pmap_nlogit_fgh!), x_initial, algorithm, optim_opts)
	else 
		out = Optim.optimize(pmap_nlogit_ll, x_initial, NelderMead(), optim_opts)
	end

	return out

end

function estimate_nlogit_serial(nl::nlogit, opt_method::Symbol, grad_type::Symbol, x_initial::Vector{Float64}, algorithm, optim_opts)

	# Define vector 
	NUMOBS = length(cl.data)

	# Optimisation functions for master process
	nlogit_ll(beta::Vector{T}) where T<:Real = sum(map(m->ll_nlogit_case(beta, m), 1:NUMOBS))

	nlogit_analytic_grad(beta::Vector{T}) where T<:Real = sum(map(m->analytic_nlogit_case(beta, m), 1:NUMOBS))

	function nlogit_analytic_fg!(F, G, beta::Vector{T}) where T<:Real
		clc = map(m->analytic_fg_nlogit_case(beta,m), 1:NUMOBS)
		if G != nothing
			G[:] = sum([y.G for y in clc])
		end
		if F != nothing
			return sum([y.F for y in clc])
		end
	end

	nlogit_grad(beta::Vector{T}) where T<:Real = sum(map(m->grad_nlogit_case(beta,m), 1:NUMOBS))

	nlogit_Hess(beta::Vector{T}) where T<:Real = sum(map(m->hessian_nlogit_case(beta,m), 1:NUMOBS))

	function nlogit_fg!(F, G, beta::Vector{T}) where T<:Real
		clc = map(m->fg_nlogit_case(beta,m), 1:NUMOBS)
		if G != nothing
			G[:] = sum([y.G for y in clc])
		end
		if F != nothing
			return sum([y.F for y in clc])
		end
	end

	function nlogit_fgh!(F, G, H, beta::Vector{T}) where T<:Real
		clc = map(m->fgh_nlogit_case(beta, m), 1:NUMOBS)
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

	# Do optimisation 
	if opt_method == :grad
		if grad_type == :analytic
			out = Optim.optimize(Optim.only_fg!(nlogit_analytic_fg!), x_initial, algorithm, optim_opts)
		else 
			out = Optim.optimize(Optim.only_fg!(nlogit_fg!), x_initial, algorithm, optim_opts)
		end
	elseif opt_method == :hess
		out = Optim.optimize(Optim.only_fgh!(nlogit_fgh!), x_initial, algorithm, optim_opts)
	else 
		out = Optim.optimize(nlogit_ll, x_initial, NelderMead(), optim_opts)
	end

	return out

end