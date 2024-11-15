import DifferentialEquations: ODEProblem, solve, DP5, PeriodicCallback, CallbackSet, terminate!, remake
import ProgressMeter: Progress, next!
import Parameters: @unpack
import Statistics: mean, std

function H_exp(ρ_soa, H)
    H_exp = zero(ComplexF64)
    for m ∈ axes(H, 1)
        for n ∈ axes(H, 2)
            ρ_re = ρ_soa.re[n,m] # transpose
            ρ_im = ρ_soa.im[n,m] # transpose
            H_re = H.re[m,n]
            H_im = H.im[m,n]
            a1 = H_re * ρ_re - H_im * ρ_im
            a2 = H_re * ρ_im + H_im * ρ_re
            H_exp += a1 + im * a2
            # H_exp += conj(H_exp)
        end
    end
    return H_exp
    # return tr(ρ_soa * H)
end
export H_exp

function force_noupdate(E_k, ds, ds_state1, ds_state2, ρ_soa)
    F = SVector(0.0, 0.0, 0.0)

    @inbounds for q ∈ 1:3
        ds_q = ds[q]
        ds_q_re = ds_q.re
        ds_q_im = ds_q.im
        ds_state1_q = ds_state1[q]
        ds_state2_q = ds_state2[q]
        for k ∈ 1:3
            E_kq = E_k[k][q]
            E_kq_re = real(E_kq)
            E_kq_im = imag(E_kq)
            F_k_re = 0.0
            F_k_im = 0.0
            for j ∈ eachindex(ds_q)
                m = ds_state1_q[j] # excited state
                n = ds_state2_q[j] # ground state
                ρ_re = ρ_soa.re[m,n]
                ρ_im = ρ_soa.im[m,n]
                d_re = ds_q_re[j]
                d_im = ds_q_im[j]
                d_im *=- 1 # take conjugate of d to make sure the Hamiltonian terms are d⋅E* + d*⋅E 
                a1 = d_re * ρ_re - d_im * ρ_im
                a2 = d_re * ρ_im + d_im * ρ_re
                F_k_re += E_kq_re * a1 - E_kq_im * a2
                F_k_im += E_kq_im * a1 + E_kq_re * a2            
            end
            F -= (im * F_k_re - F_k_im) * ê[k] # multiply by im
        end
    end

    F += conj(F)
    return real.(F)
end
export force_noupdate

function force(p, ρ, τ)
    F = SVector(0.0, 0.0, 0.0)

    @unpack fields, Γ, d, d_nnz = p
    r = p.r0 + p.v .* τ
    update_fields!(fields, r, τ)
    base_to_soa!(ρ, p.ρ_soa)
    update_eiωt!(p.eiωt, p.ω, τ)
    Heisenberg!(p.ρ_soa, p.eiωt)

    for i ∈ eachindex(fields)

        s = fields.s[i]
        # x = sqrt(s) / (2/2 * √2)  # Arian: I do not know why I included this line
        x = sqrt(s) / (2 * √2)
        k = fields.k[i]

        @inbounds for q ∈ 1:3
            # With SI units, should be x = -h * Γ * sqrt(s) / (2 * √2 * p.λ)
            ampl = k .* (x * im * fields.E[i][q])
            # Note h * Γ / λ = 2π ħ * Γ / Λ = ħ * k * Γ, so this has units units of ħ k Γ
            d_q = @view d[:,:,q]
            d_nnz_q = d_nnz[q]
            @inbounds for j ∈ d_nnz_q
                F += ampl * d_q[j] * conj(p.ρ_soa[j]) #+ conj(ampl * d_q[j] * ρ[j])
            end
        end
    end
    F += conj(F)
    return real.(F)
end
export force

function calculate_force_from_period(p, sol; force_idxs=nothing)
    """
    Integrates the force resulting from `sol` over a time period designated by `period`.
    """
    F = SVector(0.0, 0.0, 0.0)
    if isnothing(force_idxs)
        for i ∈ eachindex(sol.t)
            F += force(p, sol.u[i], sol.t[i])
        end
        return F ./ length(sol.t)
    else
        for i ∈ eachindex(force_idxs)
            F += force(p, sol.u[i], sol.t[force_idxs[i]])
        end
        return F ./ length(force_idxs)
    end
end
export calculate_force_from_period

function find_idx_for_time(time_to_find, times, backwards)
    """
    Search backwards in the array.
    """
    if backwards
        times = reverse(times)
    end
    start_time = times[1]
    found_idx = 0
    for (i, time) in enumerate(times)
        if abs(start_time - time) > time_to_find
            found_idx = i
            break
        end
    end
    if backwards
        found_idx = length(times) + 1 - found_idx
    end
    
    return found_idx
end
export find_idx_for_time

# Implement a periodic callback to reset the force each period
function reset_force!(integrator)
    force_current_period = integrator.u[end-2:end] / integrator.p.period
    force_diff = abs(norm(force_current_period) - norm(integrator.p.force_last_period))
    force_diff_rel = force_diff / norm(integrator.p.force_last_period)
    integrator.p.force_last_period = force_current_period
    
    n = length(integrator.p.states)^2
    integrator.p.populations .= integrator.u[n+1:end-3] / integrator.p.period

    force_reltol = 1e-3 # usually 1e-3
    if (force_diff_rel < force_reltol) #|| (force_diff < 1e-6)
        terminate!(integrator)
    else
        integrator.u[end-2:end] .= 0.0
        integrator.u[n+1:end-3] .= 0.0
    end
    return nothing
end
export reset_force!

""" 
    Implement a callback to stop integrating when the force settles.

    Keeps integrating the force until it does not change significantly over a time period.
"""
function integrate_to_equilibrium!(integrator)
    integrated_avg_force = integrator.u[end-2:end] / integrator.t
    force_diff = abs(norm(integrated_avg_force) - norm(integrator.p.force_last_period))
    force_diff_rel = force_diff / norm(integrated_avg_force)
    integrator.p.force_last_period = integrated_avg_force

    n = length(integrator.p.states)^2
    integrator.p.populations .= integrator.u[n+1:end-3] / integrator.p.period

    force_reltol = 1e-3 # usually 1e-3
    # println(force_diff_rel)
    if (force_diff_rel < force_reltol) #|| (force_diff < 1e-6)
        terminate!(integrator)
    end
    return nothing
end
export integrate_to_equilibrium!

function force_scan(prob::T1, scan_values::T2, prob_func!::F1, param_func::F2, output_func::F3; n_threads=Threads.nthreads()) where {T1,T2,F1,F2,F3}

    n_values = length(first(scan_values))
    batch_size = fld(n_values, n_threads)
    remainder = n_values - batch_size * n_threads
    params = zeros(Float64, n_values)
    forces = zeros(Float64, n_values)
    populations = zeros(Float64, n_values, length(prob.p.states))

    prog_bar = Progress(n_values)

    Threads.@threads for i ∈ 1:n_threads
        prob_copy = deepcopy(prob)
        # Threads.@spawn begin
            # prob_func!(_prob, scan_values, i)
        force_cb = PeriodicCallback(reset_force!, prob_copy.p.period)
        if :callback ∈ keys(prob_copy.kwargs)
            cbs = prob_copy.kwargs[:callback]
            prob_copy = remake(prob_copy, callback=CallbackSet(cbs, force_cb))
        else
            prob_copy = remake(prob_copy, callback=force_cb)
        end
        _batch_size = i <= remainder ? (batch_size + 1) : batch_size
        batch_start_idx = 1 + (i <= remainder ? (i - 1) : remainder) + batch_size * (i-1)
        batch_idxs = batch_start_idx:(batch_start_idx + _batch_size - 1)
        for j ∈ batch_idxs
            prob_j = prob_func!(prob_copy, scan_values, j)
            sol = solve(prob_j, alg=DP5())
            params[j] = param_func(prob_j, scan_values, j)
            forces[j] = output_func(prob_j.p, sol)
            prob_j.p.force_last_period = (0, 0, 0)

            populations[j,:] .= prob_j.p.populations

            next!(prog_bar)
        end
            # return
        # end
    end
    return params, forces, populations
end
export force_scan

idx_finder = x->findall.(.==(unique(x)), Ref(x) )
function average_values(params, scan_values)
    unique_params = unique(params)
    params_idxs = idx_finder(params)
    averaged_values = zeros(length(unique_params), size(scan_values, 2))
    stddev_values = zeros(length(unique_params), size(scan_values, 2))
    for (i, (param, idxs)) ∈ enumerate(zip(unique_params, params_idxs))
        param_scan_values = scan_values[idxs,:]
        averaged_values[i,:] = mean(param_scan_values, dims=1)
        stddev_values[i,:] = std(param_scan_values, dims=1)
    end
    return unique_params, averaged_values, stddev_values
end
export average_values

function force_scan_v2(prob::T1, scan_values::T2, prob_func!::F1, output_func::F2; n_threads=Threads.nthreads()) where {T1,T2,F1,F2}

    n_values = reduce(*, size(scan_values))
    batch_size = fld(n_values, n_threads)
    remainder = n_values - batch_size * n_threads
    forces = Array{SVector{3, Float64}}(undef, n_values)
    populations = zeros(Float64, n_values, length(prob.p.states))

    prog_bar = Progress(n_values)

    Threads.@threads for i ∈ 1:n_threads
        prob_copy = deepcopy(prob)
        # Threads.@spawn begin
            # prob_func!(_prob, scan_values, i)
        force_cb = PeriodicCallback(reset_force!, prob_copy.p.period)
        if :callback ∈ keys(prob_copy.kwargs)
            cbs = prob_copy.kwargs[:callback]
            prob_copy = remake(prob_copy, callback=CallbackSet(cbs, force_cb))
        else
            prob_copy = remake(prob_copy, callback=force_cb)
        end
        _batch_size = i <= remainder ? (batch_size + 1) : batch_size
        batch_start_idx = 1 + (i <= remainder ? (i - 1) : remainder) + batch_size * (i-1)
        for j ∈ batch_start_idx:(batch_start_idx + _batch_size - 1)
            prob_j = prob_func!(prob_copy, scan_values, j)
            sol = solve(prob_j, alg=DP5())
            forces[j] = output_func(prob_j.p, sol)
            prob_j.p.force_last_period = (0, 0, 0)

            populations[j,:] .= prob_j.p.populations

            next!(prog_bar)
        end
    end
    return forces, populations
end
export force_scan_v2



function force_stochastic(E_k, ds, ds_state1, ds_state2, ψ_soa, eiωt)
""" Returns the expectation value of the dipole force operator: f = <-∇H_dipole>
"""
    F = @SVector Complex{Float64}[0,0,0]

    @inbounds @fastmath for q ∈ 1:3
        ds_q = ds[q]
        ds_q_re = ds_q.re
        ds_q_im = ds_q.im
        ds_state1_q = ds_state1[q]
        ds_state2_q = ds_state2[q]
        for k ∈ 1:3
            E_kq = E_k[k][q]
            E_kq_re = real(E_kq)
            E_kq_im = imag(E_kq)
            F_k_re = 0.0
            F_k_im = 0.0
            for j ∈ eachindex(ds_q)
                m = ds_state1_q[j] # excited state
                n = ds_state2_q[j] # ground state
                
                # construct ρ_mn = c_m c_n^*
                # ρ_mn = conj(ψ_soa[n]*eiωt[n]) * ψ_soa[m]*eiωt[m]

                c_m = ψ_soa[m] * conj(eiωt[m]) # exp(-iωt) factor to transform to Heisenberg picture
                c_n = ψ_soa[n] * conj(eiωt[n]) # exp(-iωt) factor to transform to Heisenberg picture

                ρ_mn = c_m * conj(c_n)

                ρ_re = real(ρ_mn)
                ρ_im = imag(ρ_mn)
                
                d_re = ds_q_re[j]
                d_im = ds_q_im[j]

                a1 = d_re * ρ_re - d_im * ρ_im
                a2 = d_re * ρ_im + d_im * ρ_re
                F_k_re += E_kq_re * a1 - E_kq_im * a2
                F_k_im += E_kq_im * a1 + E_kq_re * a2     
            end
            F -= (im * F_k_re - F_k_im) * ê[k] # multiply by im
        end
    end
    F += conj(F)

    return real.(F)
end