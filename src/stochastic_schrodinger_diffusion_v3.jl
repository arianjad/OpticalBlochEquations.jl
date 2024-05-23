"""
    - Fixed timestepping
"""

function normalize!(p, ψ)
    ψ_norm_squared = zero(Float64)
    for i ∈ 1:p.n_states
        ψ_norm_squared += norm(ψ[i])^2
    end
    ψ_norm = sqrt(ψ_norm_squared)
    for i ∈ 1:p.n_states
        ψ[i] /= ψ_norm
    end
    return ψ_norm
end

"""
    First calculate ⟨f²⟩ - ⟨f⟩².

    ⟨f²⟩ = ⟨ψ|ff|ψ⟩
    ⟨f⟩² = ⟨ψ|f|ψ⟩²
"""
function calculate_force_variance(p_diffusion, ψ)
    @unpack p, dχ, χ = p_diffusion

    for i ∈ eachindex(χ)
        χ[i] = 0.0
    end
    fψ_to_χ!(p, ψ, χ, 1, 1)

    f_squared_exp = force_stochastic_ψ1ψ2(p, ψ, χ, 1)
    f_exp_squared = force_stochastic_ψ1ψ2(p, ψ, ψ, 1)^2

    f_σ2 = f_squared_exp - f_exp_squared

    return f_σ2
end
export calculate_force_variance

function recoil_kick(ψ, p, m)
    
    dp = sample_direction(m)
    dv = dp ./ p.mass
    ψ[p.n_states + p.n_excited + 4] += dv[1]
    ψ[p.n_states + p.n_excited + 5] += dv[2]
    ψ[p.n_states + p.n_excited + 6] += dv[3]

    return nothing
end

function recoil_kick_1D(dψ, ψ, p, m, i)
    
    sgn = sign(0.5 - rand())
    ψ[p.n_states + p.n_excited + 3 + i] += sgn * (m / p.mass)

    return nothing
end

"""
    Evolve the wavefunction ψ from time t to time t + δt, where δt is a fixed timestep.

    Additionally, evolve a wavefunction used to evaluate the diffusion.
"""
function evolve_with_diffusion!(p_diffusion, dψ, ψ, t, δt)

    p = p_diffusion.p

    # update dψ
    update_dψ!(dψ, ψ, p_diffusion, t)

    # update diffusion
    previous_f_σ2 = p_diffusion.f_σ2
    current_f_σ2 = calculate_force_variance(p_diffusion, ψ)
    p_diffusion.D = (current_f_σ2 - previous_f_σ2) / (2δt)
    p_diffusion.f_σ2 = current_f_σ2

    dp = sqrt(2abs(p_diffusion.D) * δt)

    recoil_kick_1D(dψ, ψ, p, dp, 1)
    recoil_kick_1D(dψ, ψ, p, dp, 2)
    recoil_kick_1D(dψ, ψ, p, dp, 3)

    # first check for a quantum jump, with δp = δt * [excited state population]
    δp = zero(Float64)
    for i ∈ (p.n_states - p.n_excited + 1):p.n_states
        δp += δt * norm(ψ[i])^2
    end

    if rand() < δp
        collapse!(ψ, p)

        # give one recoil kick
        recoil_kick(ψ, p, 1)

        # reset force variance
        p_diffusion.f_σ2 = 0.0
    end

    # evolve the state
    for i ∈ eachindex(ψ)
        ψ[i] += dψ[i] * δt
    end

    _ = normalize!(p, ψ)

    return nothing
end
export evolve_with_diffusion!

function update_dχ!(dψ, ψ, p, t)
    @unpack ψ_soa, dψ_soa, r, ω, fields, H, H₀, ∇H, E_k, ds, ds_state1, ds_state2, Js, eiωt, states, extra_data, mass, k, Γ = p
    
    n_states = length(states)
    n_excited = p.n_excited
    
    r = SVector(real(ψ[n_states + n_excited + 1]), real(ψ[n_states + n_excited + 2]), real(ψ[n_states + n_excited + 3]))
    
    base_to_soa!(ψ, ψ_soa)
    
    update_H!(p, t, r, fields, H, E_k, ds, ds_state1, ds_state2, Js) # molecule-light Hamiltonian in schrodinger picutre
    
    update_eiωt!(eiωt, ω, t)
    Heisenberg!(H, eiωt)  # molecule-light Hamiltonian in interation picture
    
    ∇H = p.update_H_and_∇H(H₀, p, r, t) # Zeeman and ODT hamiltonian in schrodinger picutre
    Heisenberg!(H₀, eiωt) # Zeeman and ODT Hamiltonian in interaction picture
    
    @turbo for i ∈ eachindex(H)
        H.re[i] += H₀.re[i]
        H.im[i] += H₀.im[i]
    end
    
    mul_by_im_minus!(ψ_soa)
    mul_turbo!(dψ_soa, H, ψ_soa)
    
    soa_to_base!(dψ, dψ_soa)
    
    f = force_stochastic(E_k, ds, ds_state1, ds_state2, ψ_soa, eiωt) # force due to lasers

    # accumulate excited state populations
    for i ∈ 1:n_excited
        dψ[n_states + i] = norm(ψ[n_states - n_excited + i])^2
    end
    
    for i ∈ 1:3
        dψ[n_states + n_excited + i] = ψ[n_states + n_excited + i + 3] # update position
        dψ[n_states + n_excited + 3 + i] = f[i] / mass # update velocity
        dψ[n_states + n_excited + 6 + i] = f[i] # update INTEGRATED force
    end
    
    return nothing
end
export update_dχ!

function update_dψ!(dψ, ψ, p_diffusion, t)

    p = p_diffusion.p
    @unpack ψ_soa, dψ_soa, r, ω, fields, H, H₀, ∇H, E_k, ds, ds_state1, ds_state2, Js, eiωt, states, extra_data, mass, k, Γ = p
    
    n_states = length(states)
    n_excited = p.n_excited
    
    r = SVector(real(ψ[n_states + n_excited + 1]), real(ψ[n_states + n_excited + 2]), real(ψ[n_states + n_excited + 3]))
    
    base_to_soa!(ψ, ψ_soa)
    
    update_H!(p, t, r, fields, H, E_k, ds, ds_state1, ds_state2, Js) # molecule-light Hamiltonian in schrodinger picutre
    
    update_eiωt!(eiωt, ω, t)
    Heisenberg!(H, eiωt)  # molecule-light Hamiltonian in interation picture
    
    ∇H = p.update_H_and_∇H(H₀, p, r, t) # Zeeman and ODT hamiltonian in schrodinger picutre
    Heisenberg!(H₀, eiωt) # Zeeman and ODT Hamiltonian in interaction picture
    
    @turbo for i ∈ eachindex(H)
        H.re[i] += H₀.re[i]
        H.im[i] += H₀.im[i]
    end
    
    mul_by_im_minus!(ψ_soa)
    mul_turbo!(dψ_soa, H, ψ_soa)
    
    soa_to_base!(dψ, dψ_soa)
    
    f = force_stochastic(E_k, ds, ds_state1, ds_state2, ψ_soa, eiωt) # force due to lasers

    H₀_expectation = operator_matrix_expectation(H₀, ψ_soa)
    f += ∇H .* (-H₀_expectation) # force due to conservative potential

    # # add gravity to the force
    # g = -9.81 / (Γ^2/k)
    # f += SVector{3,Float64}(0,mass*g,0)

    # accumulate excited state populations
    for i ∈ 1:n_excited
        dψ[n_states + i] = norm(ψ[n_states - n_excited + i])^2
    end
    
    for i ∈ 1:3
        dψ[n_states + n_excited + i] = ψ[n_states + n_excited + i + 3] # update position
        dψ[n_states + n_excited + 3 + i] = f[i] / mass # update velocity
        dψ[n_states + n_excited + 6 + i] = f[i] # update INTEGRATED force
    end

    # # set the instantaneous force
    # p_diffusion.Fx = f[1]
    # p_diffusion.Fy = f[2]
    # p_diffusion.Fz = f[3]
    
    return nothing
end
export update_dψ!

function collapse!(ψ, p)

    n_states = p.n_states
    n_excited = p.n_excited
    n_ground = p.n_ground
    d = p.d
    
    # A photon is observed.
    # Measure the polarization of the photon along z.
    p⁺ = 0.0
    p⁰ = 0.0
    p⁻ = 0.0
    
    for i ∈ 1:n_excited
        ψ_pop = norm(ψ[n_ground + i])^2
        for j ∈ 1:n_ground
            p⁺ += ψ_pop * norm(d[j,n_ground+i,1])^2
            p⁰ += ψ_pop * norm(d[j,n_ground+i,2])^2
            p⁻ += ψ_pop * norm(d[j,n_ground+i,3])^2
        end
        # note the polarization p in d[:,:,p] is defined to be m_e - m_g, 
        # whereas the polarization of the emitted photon is m_g - m_e
    end
    
    # set all ground state amplitudes to zero
    for i ∈ 1:n_ground
        ψ[i] = 0.0
    end
    
    p_norm = p⁺ + p⁰ + p⁻
    rn = rand() * p_norm
    pol = 0
    if 0 < rn <= p⁺ # photon is measured to have polarization σ⁺
        pol = 1
    elseif p⁺ < rn <= p⁺ + p⁰ # photon is measured to have polarization σ⁰
        pol = 2
    else # photon is measured to have polarization σ⁻
        pol = 3
    end
    
    for i in 1:n_ground
        for j in (n_ground+1):n_states
            ψ[i] += ψ[j] * d[i,j,pol]
        end
    end
    
    # zero excited state amplitudes
    for i ∈ (n_ground + 1):n_states
        ψ[i] = 0.0
    end
    
    p.n_scatters += 1
    
    # zero excited state populations
    for i ∈ (n_states+1):(n_states+n_excited)
        ψ[i] = 0.0
    end

    return nothing
end
export collapse!

"""
    Evaluate a * fψ, where f is the force operator, and place the result in χ.
"""
function fψ_to_χ!(p, ψ, χ, a, k)

    @unpack E_k, ds, ds_state1, ds_state2, eiωt = p

    @inbounds @fastmath for q ∈ 1:3

        ds_q = ds[q]
        ds_q_re = ds_q.re
        ds_q_im = ds_q.im
        ds_state1_q = ds_state1[q]
        ds_state2_q = ds_state2[q]

        E_kq = E_k[k][q]
        E_kq_re = real(E_kq)
        E_kq_im = imag(E_kq)

        for j ∈ eachindex(ds_q)
            m = ds_state1_q[j] # excited state
            n = ds_state2_q[j] # ground state

            # find the density matrix element
            c_m = conj(eiωt[m]) # exp(-iωt) factor to transform to interaction picture
            c_n = conj(eiωt[n]) # exp(-iωt) factor to transform to interaction picture

            ρ_mn = conj(c_m) * c_n

            ρ_re = real(ρ_mn)
            ρ_im = imag(ρ_mn)
            
            d_re = ds_q_re[j]
            d_im = ds_q_im[j]

            a1 = d_re * ρ_re - d_im * ρ_im
            a2 = d_re * ρ_im + d_im * ρ_re
            F_k_re = E_kq_re * a1 - E_kq_im * a2
            F_k_im = E_kq_im * a1 + E_kq_re * a2
            
            # note the factor -i
            val = -im * (F_k_re + im * F_k_im)

            χ[m] += a * val * ψ[n]
            χ[n] += a * conj(val) * ψ[m]

        end
    end
    return nothing
end

"""
    Evolve the wavefunction ψ from time t to time t + δt, where δt is a fixed timestep.
"""
function evolve!(p, dψ, ψ, t, δt)

    # update dψ
    update_dψ!(dψ, ψ, p, t)

    # first check for a quantum jump, with δp = δt * [excited state population]
    δp = zero(Float64)
    for i ∈ (p.n_states - p.n_excited + 1):p.n_states
        δp += δt * norm(ψ[i])^2
    end
    if rand() < δp
        # whenever a collapse happens, we also evaluate the diffusion coefficient
        collapse!(ψ, p)
    end

    # evolve the state
    for i ∈ eachindex(ψ)
        ψ[i] += dψ[i] * δt
    end

    _ = normalize!(p, ψ)

    return nothing
end
export evolve!

"""
    Evolve the wavefunction ψ from time t to time t + δt, where δt is a fixed timestep.
"""
function evolve_no_jumps!(p, dψ, ψ, t, δt)

    # update dψ
    update_dχ!(dψ, ψ, p, t)

    # evolve the state
    for i ∈ eachindex(ψ)
        ψ[i] += dψ[i] * δt
    end

    _ = normalize!(p, ψ)

    return nothing
end
export evolve!

"""
    Evaluate ⟨ψ₁|fₖ|ψ₂⟩
"""
function force_stochastic_ψ1ψ2(p, ψ1, ψ2, k)
    @unpack E_k, ds, ds_state1, ds_state2, eiωt = p

    F = zero(Float64)

    @inbounds @fastmath for q ∈ 1:3
        ds_q = ds[q]
        ds_q_re = ds_q.re
        ds_q_im = ds_q.im
        ds_state1_q = ds_state1[q]
        ds_state2_q = ds_state2[q]
        E_kq = E_k[k][q]
        E_kq_re = real(E_kq)
        E_kq_im = imag(E_kq)
        F_k_re = 0.0
        F_k_im = 0.0
        for j ∈ eachindex(ds_q)
            m = ds_state1_q[j] # excited state
            n = ds_state2_q[j] # ground state
            
            # construct ρ_mn = c_m c_n^*
            # ρ_mn = ψ2[m]*eiωt[m] * conj(ψ1[n]*eiωt[n]) 

            # the terms evaluated are ⟨e (e^(-iω₁t)|e⟩⟨g|e^(iω₂t)) g⟩
            c_m = conj(eiωt[m]) # exp(-iωt) factor to transform to Heisenberg picture
            c_n = conj(eiωt[n]) # exp(-iωt) factor to transform to Heisenberg picture

            ρ_mn = conj(c_m) * c_n * conj(ψ1[m]) * ψ2[n]

            ρ_re = real(ρ_mn)
            ρ_im = imag(ρ_mn)
            
            d_re = ds_q_re[j]
            d_im = ds_q_im[j]

            a1 = d_re * ρ_re - d_im * ρ_im
            a2 = d_re * ρ_im + d_im * ρ_re
            F_k_re += E_kq_re * a1 - E_kq_im * a2
            F_k_im += E_kq_im * a1 + E_kq_re * a2    
             
        end
        F -= (im * F_k_re - F_k_im) # multiply by -im
    end
    F += conj(F)

    return F
end