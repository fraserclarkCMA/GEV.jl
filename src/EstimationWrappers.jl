# Wrapper for Estimation
function estimate_clogit(cl::clogit; opt_mode = :serial, opt_method = :none, grad_type = :analytic,
						x_initial = randn(cl.model.nx), algorithm = LBFGS(), batch_size = 1, 
						optim_opts = Optim.Options(), workers=workers())

	if opt_mode == :pool_parallel 

		# Pass cl.data once. Then call subsets using integers in passdata
		pool = CachingPool(workers)

		# Pass cl.data using closure
		pool_ll_clogit_case(pd) = ll_clogit_case(pd.theta, cl.data, pd.id)
		pool_grad_clogit_case(pd) = grad_clogit_case(pd.theta, cl.data, pd.id)
		pool_analytic_grad_clogit_case(pd) = analytic_grad_clogit_case(pd.theta, cl.data, pd.id)
		pool_hessian_clogit_case(pd) = hessian_clogit_case(pd.theta, cl.data, pd.id)
		pool_fg_clogit_case(pd) = fg_clogit_case(pd.theta, cl.data, pd.id)
		pool_analytic_fg_clogit_case(pd) = analytic_fg_clogit_case(pd.theta, cl.data, pd.id)
		pool_fgh_clogit_case(pd) = fgh_clogit_case(pd.theta, cl.data, pd.id)

		function pool_pmap_clogit_ll(beta::Vector{T}) where T<:Real 
			PD = [passdata(beta, i) for i in 1:length(cl.data)]	
			sum(pmap(pool_ll_clogit_case, pool, PD, batch_size=batch_size))
		end

		function pool_pmap_clogit_analytic_grad(beta::Vector{T}) where T<:Real
			PD = [passdata(beta, i) for i in 1:length(cl.data)]	
			sum(pmap(pool_analytic_grad_clogit_case, pool, PD, batch_size=batch_size))
		end

		function pool_pmap_clogit_analytic_fg!(F, G, beta::Vector{T}) where T<:Real
			PD = [passdata(beta, i) for i in 1:length(cl.data)]	
			clc = pmap(pool_analytic_fg_clogit_case, pool, PD, batch_size=batch_size)
			if G != nothing
				G[:] = sum([y.G for y in clc])
			end
			if F != nothing
				return sum([y.F for y in clc])
			end
		end

		function pool_pmap_clogit_grad(beta::Vector{T}) where T<:Real
			PD = [passdata(beta, i) for i in 1:length(cl.data)]	
			sum(pmap(pool_grad_clogit_case, pool, PD, batch_size=batch_size))
		end

		function pool_pmap_clogit_Hess(beta::Vector{T}) where T<:Real
			PD = [passdata(beta, i) for i in 1:length(cl.data)]	
			sum(pmap(pool_hessian_clogit_case, pool, PD, batch_size=batch_size))
		end

		function pool_pmap_clogit_fg!(F, G, beta::Vector{T}) where T<:Real
			PD = [passdata(beta, i) for i in 1:length(cl.data)]	
			clc = pmap(pool_fg_clogit_case, pool, PD, batch_size=batch_size)
			if G != nothing
				G[:] = sum([y.G for y in clc])
			end
			if F != nothing
				return sum([y.F for y in clc])
			end
		end

		function pool_pmap_clogit_fgh!(F, G, H, beta::Vector{T}) where T<:Real
			PD = [passdata(beta, i) for i in 1:length(cl.data)]	
			clc = pmap(pool_fgh_clogit_case, pool, PD, batch_size=batch_size)
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

		if opt_method == :grad
			if grad_type == :analytic
				out = Optim.optimize(Optim.only_fg!(pool_pmap_clogit_analytic_fg!), x_initial, algorithm, optim_opts)
			else 
				out = Optim.optimize(Optim.only_fg!(pool_pmap_clogit_fg!), x_initial, algorithm, optim_opts)
			end
		elseif opt_method == :hess
			out = Optim.optimize(Optim.only_fgh!(pool_pmap_clogit_fgh!), x_initial, algorithm, optim_opts)
		else 
			out = Optim.optimize(pool_pmap_clogit_ll, x_initial, NelderMead(), optim_opts)
		end
	elseif opt_mode == :dist_parallel 

		dist_ll_clogit_case(dd::distdata) = ll_clogit_case(dd.theta, dd.data) 
		dist_grad_clogit_case(dd::distdata) = grad_clogit_case(dd.theta, dd.data) 
		dist_analytic_grad_clogit_case(dd::distdata) = analytic_grad_clogit_case(dd.theta, dd.data) 
		dist_fg_clogit_case(dd::distdata) = fg_clogit_case(dd.theta, dd.data)
		dist_analytic_fg_clogit_case(dd::distdata) = analytic_fg_clogit_case(dd.theta, dd.data)
		dist_hessian_clogit_case(dd::distdata) = hessian_clogit_case(dd.theta, dd.data)
		dist_fgh_clogit_case(dd::distdata) = fgh_clogit_case(dd.theta, dd.data)

		function dist_pmap_clogit_ll(beta::Vector{T}) where T<:Real 
			DD = [distdata(beta, cld) for cld in cl.data]	
			sum(pmap(dist_ll_clogit_case, DD, batch_size=batch_size))
		end

		function dist_pmap_clogit_analytic_grad(beta::Vector{T}) where T<:Real
			DD = [distdata(beta, cld) for cld in cl.data]	
			sum(pmap(dist_analytic_grad_clogit_case, DD, batch_size=batch_size))
		end

		function dist_pmap_clogit_analytic_fg!(F, G, beta::Vector{T}) where T<:Real
			DD = [distdata(beta, cld) for cld in cl.data]	
			clc = pmap(dist_analytic_fg_clogit_case, DD, batch_size=batch_size)
			if G != nothing
				G[:] = sum([y.G for y in clc])
			end
			if F != nothing
				return sum([y.F for y in clc])
			end
		end

		function dist_pmap_clogit_grad(beta::Vector{T}) where T<:Real
			DD = [distdata(beta, cld) for cld in cl.data]	
			sum(pmap(dist_grad_clogit_case, DD, batch_size=batch_size))
		end

		function dist_pmap_clogit_Hess(beta::Vector{T}) where T<:Real
			DD = [distdata(beta, cld) for cld in cl.data]	
			sum(pmap(dist_hessian_clogit_case, DD, batch_size=batch_size))
		end

		function dist_pmap_clogit_fg!(F, G, beta::Vector{T}) where T<:Real
			DD = [distdata(beta, cld) for cld in cl.data]	
			clc = pmap(dist_fg_clogit_case, DD, batch_size=batch_size)
			if G != nothing
				G[:] = sum([y.G for y in clc])
			end
			if F != nothing
				return sum([y.F for y in clc])
			end
		end

		function dist_pmap_clogit_fgh!(F, G, H, beta::Vector{T}) where T<:Real
			DD = [distdata(beta, cld) for cld in cl.data]	
			clc = pmap(dist_fgh_clogit_case, DD, batch_size=batch_size)
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

		if opt_method == :grad
			if grad_type == :analytic
				out = Optim.optimize(Optim.only_fg!(dist_pmap_clogit_analytic_fg!), x_initial, algorithm, optim_opts)
			else 
				out = Optim.optimize(Optim.only_fg!(dist_pmap_clogit_fg!), x_initial, algorithm, optim_opts)
			end
		elseif opt_method == :hess
			out = Optim.optimize(Optim.only_fgh!(dist_pmap_clogit_fgh!), x_initial, algorithm, optim_opts)
		else 
			out = Optim.optimize(dist_pmap_clogit_ll, x_initial, NelderMead(), optim_opts)
		end

	elseif opt_mode == :serial

		serial_ll_clogit_case(dd::distdata) = ll_clogit_case(dd.theta, dd.data) 
		serial_grad_clogit_case(dd::distdata) = grad_clogit_case(dd.theta, dd.data) 
		serial_analytic_grad_clogit_case(dd::distdata) = analytic_grad_clogit_case(dd.theta, dd.data) 
		serial_fg_clogit_case(dd::distdata) = fg_clogit_case(dd.theta, dd.data)
		serial_analytic_fg_clogit_case(dd::distdata) = analytic_fg_clogit_case(dd.theta, dd.data)
		serial_hessian_clogit_case(dd::distdata) = hessian_clogit_case(dd.theta, dd.data)
		serial_fgh_clogit_case(dd::distdata) = fgh_clogit_case(dd.theta, dd.data)

		function map_clogit_ll(beta::Vector{T}) where T<:Real 
			DD = [distdata(beta, cld) for cld in cl.data]	
			sum(map(serial_ll_clogit_case, DD))
		end

		function map_clogit_analytic_grad(beta::Vector{T}) where T<:Real
			DD = [distdata(beta, cld) for cld in cl.data]	
			sum(map(serial_analytic_grad_clogit_case, DD))
		end

		function map_clogit_analytic_fg!(F, G, beta::Vector{T}) where T<:Real
			DD = [distdata(beta, cld) for cld in cl.data]	
			clc = map(serial_analytic_fg_clogit_case, DD)
			if G != nothing
				G[:] = sum([y.G for y in clc])
			end
			if F != nothing
				return sum([y.F for y in clc])
			end
		end

		function map_clogit_grad(beta::Vector{T}) where T<:Real
			DD = [distdata(beta, cld) for cld in cl.data]	
			sum(map(serial_grad_clogit_case, DD))
		end

		function map_clogit_Hess(beta::Vector{T}) where T<:Real
			DD = [distdata(beta, cld) for cld in cl.data]	
			sum(map(serial_hessian_clogit_case, DD))
		end

		function map_clogit_fg!(F, G, beta::Vector{T}) where T<:Real
			DD = [distdata(beta, cld) for cld in cl.data]	
			clc = map(serial_fg_clogit_case, DD)
			if G != nothing
				G[:] = sum([y.G for y in clc])
			end
			if F != nothing
				return sum([y.F for y in clc])
			end
		end

		function map_clogit_fgh!(F, G, H, beta::Vector{T}) where T<:Real
			DD = [distdata(beta, cld) for cld in cl.data]	
			YY = map(serial_fgh_clogit_case, DD)
			if H != nothing
				H[:] = sum([yy.H for yy in YY])
			end
			if G != nothing
				G[:] = sum([yy.G for yy in YY])
			end
			if F != nothing
				return sum([yy.F for yy in YY])
			end
		end

		if opt_method == :grad
			if grad_type == :analytic
				out = Optim.optimize(Optim.only_fg!(map_clogit_analytic_fg!), x_initial, algorithm, optim_opts)
			else 
				out = Optim.optimize(Optim.only_fg!(map_clogit_fg!), x_initial, algorithm, optim_opts)
			end		
		elseif opt_method == :hess
			out = Optim.optimize(Optim.only_fgh!(map_clogit_fgh!), x_initial, algorithm, optim_opts)
		else 
			out = Optim.optimize(map_clogit_ll, x_initial, NelderMead(), optim_opts)
		end
	else 
		out = Optim.optimize(x->ll_clogit(x, cl.data), x_initial, algorithm, optim_opts; autodiff=:forward )
	end

	return out
end



function estimate_nlogit(nl::nlogit; opt_mode = :serial, opt_method = :none, grad_type = :analytic,
							x_initial = randn(nl.model.nx), algorithm = LBFGS(), batch_size=1,
							optim_opts = Optim.Options(), workers=workers())
	
	if opt_mode == :shared_parallel
		pool = CachingPool(workers)

		# Pass cl.data using closure
		pool_ll_nlogit_case(pd) = ll_nlogit_case(pd.theta, nl.model, nl.data, pd.id)
		pool_grad_nlogit_case(pd) = grad_nlogit_case(pd.theta,  nl.model, nl.data, pd.id)
		pool_analytic_grad_nlogit_case(pd) = analytic_grad_nlogit_case(pd.theta,  nl.model, nl.data, pd.id)
		pool_hessian_nlogit_case(pd) = hessian_nlogit_case(pd.theta, nl.model, nl.data, pd.id)
		pool_fg_clogit_case(pd) = fg_nlogit_case(pd.theta, nl.model, nl.data, pd.id)
		pool_analytic_fg_nlogit_case(pd) = analytic_fg_nlogit_case(pd.theta, nl.model, nl.data, pd.id)
		pool_fgh_nlogit_case(pd) = fgh_nlogit_case(pd.theta, nl.model, nl.data, pd.id)

		function pool_pmap_nlogit_ll(theta::Vector{T}) where T<:Real 
			PD = [passdata(theta, i) for i in 1:length(nl.data)]	
			sum(pmap(pool_ll_nlogit_case, pool, PD, batch_size=batch_size))
		end

		function pool_pmap_nlogit_analytic_grad(theta::Vector{T}) where T<:Real
			PD = [passdata(theta, i) for i in 1:length(nl.data)]	
			sum(pmap(pool_analytic_grad_clogit_case, pool, PD, batch_size=batch_size))
		end

		function pool_pmap_nlogit_analytic_fg!(F, G, theta::Vector{T}) where T<:Real
			PD = [passdata(theta, i) for i in 1:length(data)]	
			nlc = pmap(pool_analytic_fg_nlogit_case, pool, PD, batch_size=batch_size)
			if G != nothing
				G[:] = sum([y.G for y in nlc])
			end
			if F != nothing
				return sum([y.F for y in nlc])
			end
		end

		function pool_pmap_nlogit_grad(theta::Vector{T}) where T<:Real
			PD = [passdata(theta, i) for i in 1:length(nl.data)]	
			sum(pmap(pool_grad_clogit_case, pool, PD, batch_size=batch_size))
		end

		function pool_pmap_nlogit_Hess(theta::Vector{T}) where T<:Real
			PD = [passdata(theta, i) for i in 1:length(nl.data)]	
			sum(pmap(pool_hessian_nlogit_case, pool, PD, batch_size=batch_size))
		end

		function pool_pmap_nlogit_fg!(F, G, theta::Vector{T}) where T<:Real
			PD = [passdata(theta, i) for i in 1:length(nl.data)]	
			nlc = pmap(pool_fg_nlogit_case, pool, PD, batch_size=batch_size)
			if G != nothing
				G[:] = sum([y.G for y in nlc])
			end
			if F != nothing
				return sum([y.F for y in nlc])
			end
		end

		function pool_pmap_nlogit_fgh!(F, G, H, theta::Vector{T}) where T<:Real
			PD = [passdata(theta, i) for i in 1:length(nl.data)]	
			nlc = pmap(pool_fgh_nlogit_case, pool, PD, batch_size=batch_size)
			if H != nothing
				H[:] = sum([y.H for y in nlc])
			end
			if G != nothing
				G[:] = sum([y.G for y in nlc])
			end
			if F != nothing
				return sum([y.F for y in nlc])
			end
		end

		if opt_method == :grad
			if grad_type == :analytic
				out = Optim.optimize(Optim.only_fg!(pool_pmap_nlogit_analytic_fg!), x_initial, algorithm, optim_opts)
			else 
				out = Optim.optimize(Optim.only_fg!(pool_pmap_nlogit_fg!), x_initial, algorithm, optim_opts)
			end
		elseif opt_method == :hess
			out = Optim.optimize(Optim.only_fgh!(pool_pmap_nlogit_fgh!), x_initial, algorithm, optim_opts)
		else 
			out = Optim.optimize(pool_pmap_nlogit_ll, x_initial, NelderMead(), optim_opts)
		end
	elseif opt_mode == :dist_parallel 
		
		dist_ll_nlogit_case(dd::distdata) = ll_clogit_case(dd.theta, nl.model, dd.data) 
		dist_grad_nlogit_case(dd::distdata) = grad_clogit_case(dd.theta, nl.model, dd.data) 
		dist_analytic_grad_nlogit_case(dd::distdata) = analytic_grad_clogit_case(dd.theta, nl.model, dd.data) 
		dist_fg_nlogit_case(dd::distdata) = fg_clogit_case(dd.theta, nl.model, dd.data)
		dist_analytic_fg_nlogit_case(dd::distdata) = analytic_fg_clogit_case(dd.theta, nl.model, dd.data)
		dist_hessian_nlogit_case(dd::distdata) = hessian_clogit_case(dd.theta, nl.model, dd.data)
		dist_fgh_nlogit_case(dd::distdata) = fgh_clogit_case(dd.theta, nl.model, dd.data)
	
		function dist_pmap_nlogit_ll(theta::Vector{T}) where T<:Real 
			DD = [distdata(theta, nld) for nld in nl.data]	
			sum(pmap(dist_ll_nlogit_case, DD, batch_size=batch_size))
		end

		function dist_pmap_nlogit_analytic_grad(theta::Vector{T}) where T<:Real
			DD = [distdata(theta, nld) for nld in nl.data]	
			sum(pmap(dist_analytic_grad_clogit_case, DD, batch_size=batch_size))
		end

		function dist_pmap_nlogit_analytic_fg!(F, G, theta::Vector{T}) where T<:Real
			DD = [distdata(theta, nld) for nld in nl.data]	
			nlc = pmap(dist_analytic_fg_nlogit_case, DD, batch_size=batch_size)
			if G != nothing
				G[:] = sum([y.G for y in nlc])
			end
			if F != nothing
				return sum([y.F for y in nlc])
			end
		end

		function dist_pmap_nlogit_grad(theta::Vector{T}) where T<:Real
			DD = [distdata(theta, nld) for nld in nl.data]	
			sum(pmap(dist_grad_clogit_case, DD, batch_size=batch_size))
		end

		function dist_pmap_nlogit_Hess(theta::Vector{T}) where T<:Real
			DD = [distdata(theta, nld) for nld in nl.data]	
			sum(pmap(dist_hessian_nlogit_case, DD, batch_size=batch_size))
		end

		function dist_pmap_nlogit_fg!(F, G, theta::Vector{T}) where T<:Real
			DD = [distdata(theta, nld) for nld in nl.data]	
			nlc = pmap(dist_fg_nlogit_case, DD, batch_size=batch_size)
			if G != nothing
				G[:] = sum([y.G for y in nlc])
			end
			if F != nothing
				return sum([y.F for y in nlc])
			end
		end

		function dist_pmap_nlogit_fgh!(F, G, H, theta::Vector{T}) where T<:Real
			DD = [distdata(theta, nld) for nld in nl.data]	
			nlc = pmap(dist_fgh_nlogit_case, DD, batch_size=batch_size)
			if H != nothing
				H[:] = sum([y.H for y in nlc])
			end
			if G != nothing
				G[:] = sum([y.G for y in nlc])
			end
			if F != nothing
				return sum([y.F for y in nlc])
			end
		end

		if opt_method == :grad
			if grad_type == :analytic
				out = Optim.optimize(Optim.only_fg!(dist_pmap_nlogit_analytic_fg!), x_initial, algorithm, optim_opts)
			else 
				out = Optim.optimize(Optim.only_fg!(dist_pmap_nlogit_fg!), x_initial, algorithm, optim_opts)
			end
		elseif opt_method == :hess
			out = Optim.optimize(Optim.only_fgh!(dist_pmap_nlogit_fgh!), x_initial, algorithm, optim_opts)
		else 
			out = Optim.optimize(dist_pmap_nlogit_ll, x_initial, NelderMead(), optim_opts)
		end

	elseif opt_mode == :serial 

		clos_ll_nlogit_case(pd) = ll_nlogit_case(pd.theta, nl, pd.id)
		clos_grad_nlogit_case(pd) = grad_nlogit_case(pd.theta, nl, pd.id)
		clos_analytic_grad_nlogit_case(pd) = analytic_grad_nlogit_case(pd.theta, nl, pd.id)
		clos_hessian_nlogit_case(pd) = hessian_nlogit_case(pd.theta, nl, pd.id)
		clos_fg_nlogit_case(pd) = fg_nlogit_case(pd.theta, nl, pd.id)
		clos_analytic_fg_nlogit_case(pd) = analytic_fg_nlogit_case(pd.theta, nl, pd.id)
		clos_fgh_nlogit_case(pd) = fgh_nlogit_case(pd.theta, nl, pd.id)
		
		function map_nlogit_ll(theta::Vector{T}) where T<:Real 
			PD = [passdata(theta, i) for i in 1:length(nl.data)]	
			sum(map(clos_ll_nlogit_case, PD))
		end

		function map_nlogit_analytic_grad(theta::Vector{T}) where T<:Real
			PD = [passdata(theta, i) for i in 1:length(nl.data)]	
			sum(map(clos_analytic_grad_nlogit_case, PD))
		end
		
		function map_nlogit_analytic_fg!(F, G, theta::Vector{T}) where T<:Real
			PD = [passdata(theta, i) for i in 1:length(nl.data)]	
			nlc = map(clos_analytic_fg_nlogit_case, PD)
			if G != nothing
				G[:] = sum([y.G for y in nlc])
			end
			if F != nothing
				return sum([y.F for y in nlc])
			end
		end

		function map_nlogit_grad(theta::Vector{T}) where T<:Real
			PD = [passdata(theta, i) for i in 1:length(nl.data)]	
			sum(map(clos_grad_nlogit_case, PD))
		end

		function map_nlogit_fg!(F, G, theta::Vector{T}) where T<:Real
			PD = [passdata(theta, i) for i in 1:length(nl.data)]	
			nlc = map(clos_fg_nlogit_case, PD)
			if G != nothing
				G[:] = sum([y.G for y in nlc])
			end
			if F != nothing
				return sum([y.F for y in nlc])
			end
		end

		function map_nlogit_Hess(theta::Vector{T}) where T<:Real
			PD = [passdata(theta, i) for i in 1:length(nl.data)]	
			sum(map(clos_hessian_nlogit_case, PD))
		end

		function map_nlogit_fgh!(F, G, H, theta::Vector{T}) where T<:Real
			PD = [passdata(theta, i) for i in 1:length(nl.data)]	
			nlc = map(clos_fgh_nlogit_case, PD)
			if H != nothing
				H[:] = sum([y.H for y in nlc])
			end
			if G != nothing
				G[:] = sum([y.G for y in nlc])
			end
			if F != nothing
				return sum([y.F for y in nlc])
			end
		end

		if opt_method == :grad
			if grad_type == :analytic
				out = Optim.optimize(Optim.only_fg!(map_nlogit_analytic_fg!), x_initial, algorithm, optim_opts)
			else 
				out = Optim.optimize(Optim.only_fg!(map_nlogit_fg!), x_initial, algorithm, optim_opts)
			end
		elseif opt_method == :hess
			out = Optim.optimize(Optim.only_fgh!(map_nlogit_fgh!), x_initial, algorithm, optim_opts)
		else 
			out = Optim.optimize(map_nlogit_ll, x_initial, NelderMead(), optim_opts)
		end

	else
		out = Optim.optimize(x->ll_nlogit(x, nl.data), x_initial, algorithm, optim_opts; autodiff=:forward )
	end 

	return out
end
