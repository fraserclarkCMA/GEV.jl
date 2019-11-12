# Wrapper for Estimation
function estimate_clogit(cl::clogit; opt_mode = :serial, opt_method = :none, grad_type = :analytic,
						x_initial = randn(cl.model.nx), algorithm = LBFGS(), batch_size = 1, optim_opts = Optim.Options())

	if opt_mode == :parallel 

		out = estimate_clogit_parallel(cl, opt_method, grad_type, x_initial, algorithm, batch_size, optim_opts)

	else

		out = estimate_clogit_serial(cl, opt_method, grad_type, x_initial, algorithm , optim_opts)

	end 

	return out 

end 

# Wrappers for Estimation assuming nl -> Main.nl on worker - useful in Parallel 
# Note caching pool unique to pmap call, i want multiple pmap calls so this should be more efficient despite global
ll_clogit_case(beta::Vector{T}, id::Int64) where T<:Real = ll_clogit_case(beta, Main.cl.data[id])
grad_clogit_case(beta::Vector{T}, id::Int64) where T<:Real = grad_clogit_case(beta, Main.cl.data[id]) 
hessian_clogit_case(beta::Vector{T}, id::Int64) where T<:Real = hessian_clogit_case(beta, Main.cl.data[id]) 
analytic_grad_clogit_case(beta::Vector{T}, id::Int64) where T<:Real = analytic_grad_clogit_case(beta, Main.cl.data[id]) 
fg_clogit_case(beta::Vector{T}, id::Int64) where T<:Real = fg_clogit_case(beta, Main.cl.data[id]) 
analytic_fg_clogit_case(beta::Vector{T}, id::Int64) where T<:Real = analytic_fg_clogit_case(beta, Main.cl.data[id]) 
fgh_clogit_case(beta::Vector{T}, id::Int64) where T<:Real = fgh_clogit_case(beta, Main.cl.data[id]) 

function estimate_clogit_parallel(cl::clogit, opt_method::Symbol, grad_type::Symbol, x_initial::Vector{Float64},
										 algorithm, batch_size::Int64, optim_opts)

	
	# Step 1: Copy the data to Main.cl_data on all workers
	printstyled("\nTransferring data to workers....\n"; bold=true, color=:blue)
	sendto(workers(), cl = cl)
	printstyled("Transfer of data to workers compelete\n"; bold=true, color=:blue)
	
	# Step 2: Setup
	NUMOBS = length(cl.data)

	# Optimisation functions for master process
	pmap_clogit_ll(beta::Vector{T}) where T<:Real = 
		sum(pmap(m->ll_clogit_case(beta, m), 1:NUMOBS, batch_size=batch_size))

	pmap_clogit_analytic_grad(beta::Vector{T}) where T<:Real =
		sum(pmap(m->analytic_clogit_case(beta, m), 1:NUMOBS, batch_size=batch_size))

	function pmap_clogit_analytic_fg!(F, G, beta::Vector{T}) where T<:Real
		clc = pmap(m->analytic_fg_clogit_case(beta,m), 1:NUMOBS, batch_size=batch_size)
		if G != nothing
			G[:] = sum([y.G for y in clc])
		end
		if F != nothing
			return sum([y.F for y in clc])
		end
	end

	pmap_clogit_grad(beta::Vector{T}) where T<:Real = 
		sum(pmap(m->grad_clogit_case(beta,m), 1:NUMOBS, batch_size=batch_size))

	pmap_clogit_Hess(beta::Vector{T}) where T<:Real =
		sum(pmap(m->hessian_clogit_case(beta,m), 1:NUMOBS, batch_size=batch_size))

	function pmap_clogit_fg!(F, G, beta::Vector{T}) where T<:Real
		clc = pmap(m->fg_clogit_case(beta,m), 1:NUMOBS, batch_size=batch_size)
		if G != nothing
			G[:] = sum([y.G for y in clc])
		end
		if F != nothing
			return sum([y.F for y in clc])
		end
	end

	function pmap_clogit_fgh!(F, G, H, beta::Vector{T}) where T<:Real
		clc = pmap(m->fgh_clogit_case(beta, m), 1:NUMOBS, batch_size=batch_size)
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
			out = Optim.optimize(Optim.only_fg!(pmap_clogit_analytic_fg!), x_initial, algorithm, optim_opts)
		else 
			out = Optim.optimize(Optim.only_fg!(pmap_clogit_fg!), x_initial, algorithm, optim_opts)
		end
	elseif opt_method == :hess
		out = Optim.optimize(Optim.only_fgh!(pmap_clogit_fgh!), x_initial, algorithm, optim_opts)
	else 
		out = Optim.optimize(pmap_clogit_ll, x_initial, NelderMead(), optim_opts)
	end

	return out

end

function estimate_clogit_serial(cl::clogit, opt_method::Symbol, grad_type::Symbol, x_initial::Vector{Float64}, algorithm, optim_opts)

	# Define vector 
	NUMOBS = length(cl.data)

	# Optimisation functions for master process
	clogit_ll(beta::Vector{T}) where T<:Real = sum(map(m->ll_clogit_case(beta, m), 1:NUMOBS))

	clogit_analytic_grad(beta::Vector{T}) where T<:Real = sum(map(m->analytic_clogit_case(beta, m), 1:NUMOBS))

	function clogit_analytic_fg!(F, G, beta::Vector{T}) where T<:Real
		clc = map(m->analytic_fg_clogit_case(beta,m), 1:NUMOBS)
		if G != nothing
			G[:] = sum([y.G for y in clc])
		end
		if F != nothing
			return sum([y.F for y in clc])
		end
	end

	clogit_grad(beta::Vector{T}) where T<:Real = sum(map(m->grad_clogit_case(beta,m), 1:NUMOBS))

	clogit_Hess(beta::Vector{T}) where T<:Real = sum(map(m->hessian_clogit_case(beta,m), 1:NUMOBS))

	function clogit_fg!(F, G, beta::Vector{T}) where T<:Real
		clc = map(m->fg_clogit_case(beta,m), 1:NUMOBS)
		if G != nothing
			G[:] = sum([y.G for y in clc])
		end
		if F != nothing
			return sum([y.F for y in clc])
		end
	end

	function clogit_fgh!(F, G, H, beta::Vector{T}) where T<:Real
		clc = map(m->fgh_clogit_case(beta, m), 1:NUMOBS)
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
			out = Optim.optimize(Optim.only_fg!(clogit_analytic_fg!), x_initial, algorithm, optim_opts)
		else 
			out = Optim.optimize(Optim.only_fg!(clogit_fg!), x_initial, algorithm, optim_opts)
		end
	elseif opt_method == :hess
		out = Optim.optimize(Optim.only_fgh!(clogit_fgh!), x_initial, algorithm, optim_opts)
	else 
		out = Optim.optimize(clogit_ll, x_initial, NelderMead(), optim_opts)
	end

	return out

end