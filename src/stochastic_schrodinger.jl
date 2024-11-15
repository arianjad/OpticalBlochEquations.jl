using Distributions, StatsBase

function SE_collapse!(integrator)
    """ Periodically called by the solver. Random spontaneous emission. Restart excited state population averaging. """
   
    rn = rand()  # sample from random variable uniform([0,1]).
    
    n_states = length(integrator.p.states)
    n_excited = integrator.p.n_excited
    
    # excited state population integrated over dT  
    lower_baseline = 0.0
    upper_baseline = 0.0
    
    for i in 1:n_excited
        upper_baseline += norm(integrator.u[n_states + i])
        if lower_baseline <= rn < upper_baseline

            i_ground, δm = spontaneous_emission_event(integrator.p, i+n_states-n_excited)
            
            # reset state to i_ground
            for i in 1:n_states
                integrator.u[i] = 0.0 #-= integrator.u[i]
            end
            
            integrator.u[i_ground] = 1

            dp = sample_direction(1)
            dv = dp ./ integrator.p.mass
            integrator.u[n_states + n_excited + 4] += dv[1]
            integrator.u[n_states + n_excited + 5] += dv[2]
            integrator.u[n_states + n_excited + 6] += dv[3]
            
            integrator.p.n_scatters += 1
            break
        end
        lower_baseline += norm(integrator.u[n_states + i])
    end
    
    # reset excited state population accumulation
    integrator.u[n_states + 1: n_states + n_excited] .= 0
    
    for i in 1:n_states
        integrator.p.populations[i] = norm(integrator.u[i])^2
    end
end
export SE_collapse!

function SE_collapse_pol!(integrator)
    """ 
    The previous method collapses both the excited state and the polarization of the emitted photon.
    But in reality, we should collapse something iff the information is leaked into the environment, which, in our case
    does NOT include the identity of the excited state, no matter what you choose to measure.
    
    The current approach assumes the environment measures the polarization of the photon along z. We do not collapse which 
    excited state the molecule is in, therefore preserving some coherence (the state after collapse can be a superposition
    of different ground states).
    
    And because phase information between excited states is preserved during collapse, we can not sample based on average 
    excited state population anymore.
    
    See Dalibard, Castin & Molmer PRL 1992
    """ 
    p = integrator.p
    n_states = p.n_states
    n_excited = p.n_excited
    n_ground = p.n_ground
    dT = p.dT
    d = p.d
    ψ = integrator.u
    
    # save_every = 10000, save_counter = 0, trajectory = []
    p.save_counter += 1
    if p.save_counter >= p.save_every
        push!(p.trajectory, deepcopy(integrator.u))
        p.save_counter = 0
    end
    
    p_decay = 0.0
    for i in 1:n_excited
        p_decay += norm(ψ[n_ground + i])^2 * dT
    end
    
    rn = rand()
    if rn > p_decay
        # No photon is observed. Coherent population by H_eff.
        return nothing
    end
    
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
    
    p_norm = p⁺ + p⁰ + p⁻
    rn = rand() * p_norm
    for i ∈ 1:n_ground
        ψ[i] = 0.0
    end
    
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

    # zero excited state populations
    for i ∈ (n_states + 1):n_excited
        ψ[i] = 0.0
    end
    
    ψ_norm = 0.0
    for i ∈ 1:n_states
        ψ_norm += norm(ψ[i])^2
    end
    ψ_norm = sqrt(ψ_norm)
    for i ∈ 1:n_states
        ψ[i] /= ψ_norm
    end
    
    p.n_scatters += 1
    
     # reset excited state population accumulation
#     integrator.u[n_states + 1: n_states + n_excited] .= 0
    
#     for i in 1:n_states
#         integrator.p.populations[i] = norm(integrator.u[i])^2
#     end
    
    dp = sample_direction(1)
    dv = dp ./ p.mass
    integrator.u[n_states + n_excited + 4] += dv[1]
    integrator.u[n_states + n_excited + 5] += dv[2]
    integrator.u[n_states + n_excited + 6] += dv[3]
    
    return nothing
end
export SE_collapse_pol!

function SE_collapse_pol_always_v2!(integrator)

    p = integrator.p
    n_states = p.n_states
    n_excited = p.n_excited
    n_ground = p.n_ground
    d = p.d
    ψ = integrator.u
    
    # A photon is observed.
    # Measure the polarization of the photon along z.
    p⁺ = 0.0
    p⁰ = 0.0
    p⁻ = 0.0
    
    @inbounds @fastmath for i ∈ 1:n_excited
        ψ_i = ψ[n_ground+i]
        ψ_pop = real(ψ_i)^2 + imag(ψ_i)^2
        for j ∈ 1:n_ground
            p⁺ += ψ_pop * norm(d[j,n_ground+i,1])^2
            p⁰ += ψ_pop * norm(d[j,n_ground+i,2])^2
            p⁻ += ψ_pop * norm(d[j,n_ground+i,3])^2
        end
        # note the polarization p in d[:,:,p] is defined to be m_e - m_g, 
        # whereas the polarization of the emitted photon is m_g - m_e
    end
    
    p_norm = p⁺ + p⁰ + p⁻
    rn = rand() * p_norm
    @inbounds @fastmath for i ∈ 1:n_ground
        ψ[i] = 0.0
    end
    
    pol = 0
    if 0 < rn <= p⁺ # photon is measured to have polarization σ⁺
        pol = 1
    elseif p⁺ < rn <= p⁺ + p⁰ # photon is measured to have polarization σ⁰
        pol = 2
    else # photon is measured to have polarization σ⁻
        pol = 3
    end
    
    @inbounds @fastmath for i in 1:n_ground
        for j in (n_ground+1):n_states
            ψ[i] += ψ[j] * d[i,j,pol]
        end
    end
    
    # zero excited state amplitudes
    @inbounds @fastmath for i ∈ (n_ground + 1):n_states
        ψ[i] = 0.0
    end

    ψ_norm = 0.0
    @inbounds @fastmath for i ∈ 1:n_states
        ψ_norm += real(ψ[i])^2 + imag(ψ[i])^2
    end
    ψ_norm = sqrt(ψ_norm)
    @inbounds @fastmath for i ∈ 1:n_states
        ψ[i] /= ψ_norm
    end
    
    p.n_scatters += 1
    
    # zero excited state populations
    @inbounds @fastmath for i ∈ (n_states+1):(n_states+n_excited)
        integrator.u[i] = 0.0
    end
    
    dp = sample_direction(1)
    dv = dp ./ p.mass
    integrator.u[n_states + n_excited + 4] += 2 * dv[1]
    integrator.u[n_states + n_excited + 5] += 2 * dv[2]
    integrator.u[n_states + n_excited + 6] += 2 * dv[3]
    
    p.time_to_decay = rand(p.decay_dist)

    return nothing
end
export SE_collapse_pol_always_v2!

function SE_collapse_pol_always!(integrator)

    p = integrator.p
    n_states = p.n_states
    n_excited = p.n_excited
    n_ground = p.n_ground
    d = p.d
    ψ = integrator.u
    
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
    
    p_norm = p⁺ + p⁰ + p⁻
    rn = rand() * p_norm
    for i ∈ 1:n_ground
        ψ[i] = 0.0
    end
    
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
    
    # is this required?
    ψ_norm = 0.0
    for i ∈ 1:n_states
        ψ_norm += norm(ψ[i])^2
    end
    ψ_norm = sqrt(ψ_norm)
    for i ∈ 1:n_states
        ψ[i] /= ψ_norm
    end
    
    p.n_scatters += 1
    
    # zero excited state populations
    for i ∈ (n_states+1):(n_states+n_excited)
        integrator.u[i] = 0.0
    end

    # # add a repumper photon kick along the x-axis
    # repump_states_prob = 1/150 # 1/150 photons are repumped with a laser that is not retroreflected
    # if rand() < repump_states_prob
    #     integrator.u[n_states + n_excited + 4] += 2 / p.mass # 2 photons are required to repump on average
    # end

    # repump_states_prob = 1/20 # 1/20 photons are not (000), and on average two photon scatters are required to repump
    # if rand() < repump_states_prob
    #     dp = sample_direction(1)
    #     dv = dp ./ p.mass
    #     integrator.u[n_states + n_excited + 4] += dv[1]
    #     integrator.u[n_states + n_excited + 5] += dv[2]
    #     integrator.u[n_states + n_excited + 6] += dv[3]
        
    #     dp = sample_direction(1)
    #     dv = dp ./ p.mass
    #     integrator.u[n_states + n_excited + 4] += dv[1]
    #     integrator.u[n_states + n_excited + 5] += dv[2]
    #     integrator.u[n_states + n_excited + 6] += dv[3]
    # end

     # reset excited state population accumulation
    # integrator.u[n_states + 1:n_states + n_excited] .= 0
    
#     for i in 1:n_states
#         integrator.p.populations[i] = norm(integrator.u[i])^2
#     end
    
    # add SE recoil
    dp = sample_direction(1)
    dv = dp ./ p.mass
    integrator.u[n_states + n_excited + 4] += dv[1]
    integrator.u[n_states + n_excited + 5] += dv[2]
    integrator.u[n_states + n_excited + 6] += dv[3]

    time_before_decay = integrator.t - p.last_decay_time
    for i ∈ 1:3
        integrator.u[p.n_states + p.n_excited + 3 + i] += rand((-1,1)) * sqrt( 2p.diffusion_constant * time_before_decay ) / p.mass
    end
    p.last_decay_time = integrator.t

    p.time_to_decay = rand(p.decay_dist)

    return nothing
end
export SE_collapse_pol_always!

function SE_collapse_pol_custom!(integrator)

    p = integrator.p
    n_states = p.n_states
    n_excited = p.n_excited
    n_ground = p.n_ground
    d = p.d
    ψ = integrator.u
    
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
    
    p_norm = p⁺ + p⁰ + p⁻
    rn = rand() * p_norm
    for i ∈ 1:n_ground
        ψ[i] = 0.0
    end
    
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
    
    # is this required?
    ψ_norm = 0.0
    for i ∈ 1:n_states
        ψ_norm += norm(ψ[i])^2
    end
    ψ_norm = sqrt(ψ_norm)
    for i ∈ 1:n_states
        ψ[i] /= ψ_norm
    end
    
    p.n_scatters += 1
    
    # zero excited state populations
    for i ∈ (n_states+1):(n_states+n_excited)
        integrator.u[i] = 0.0
    end

    # # add a repumper photon kick along the x-axis
    repump_states_prob = 1/150 # 1/150 photons are repumped with a laser that is not retroreflected
    if rand() < repump_states_prob
        integrator.u[n_states + n_excited + 4] += 2 / p.mass # 2 photons are required to repump on average
    end

    
    
    
    # photon recoils with custum recoil momentum (equivalent to a custum number of recoils)
    
    s = integrator.p.sim_params.s_total
    C3 = integrator.p.sim_params.C3
    C5 = integrator.p.sim_params.C5
    C7 = integrator.p.sim_params.C7
    threshold_1 = s^(1/2) / (s^(1/2) + 1/factorial(3) * C3 * s^(3/2) + 1/factorial(5) * C5 * s^(5/2) + 1/factorial(7) * C7 * s^(7/2))
    threshold_3 = threshold_1 + 1/6 * C3 * s^(3/2) / (s^(1/2) + 1/factorial(3) * C3 * s^(3/2) + 1/factorial(5) * C5 * s^(5/2) + 1/factorial(7) * C7 * s^(7/2))
    threshold_5 = threshold_3 + 1/120 * C5 * s^(5/2) / (s^(1/2) + 1/factorial(3) * C3 * s^(3/2) + 1/factorial(5) * C5 * s^(5/2) + 1/factorial(7) * C7 * s^(7/2))
    n_rand = rand()
    n_kicks = 1
    if n_rand < threshold_1
        n_kicks = 1
    elseif n_rand < threshold_3
        n_kicks = 3
    elseif n_rand < threshold_5
        n_kicks = 5
    else
        n_kicks = 7
    end
    
    
    for _ ∈ 1:n_kicks + 1
        dp = sample_direction(1)
        dv = dp ./ p.mass
        integrator.u[n_states + n_excited + 4] += dv[1]
        integrator.u[n_states + n_excited + 5] += dv[2]
        integrator.u[n_states + n_excited + 6] += dv[3]
    end

    p.time_to_decay = rand(p.decay_dist)

    return nothing
end
export SE_collapse_pol_custom!

function SE_collapse_pol_always_repump!(integrator, count_scatter=true, number_recoils=1)

    p = integrator.p
    n_states = p.n_states
    n_excited = p.n_excited
    n_ground = p.n_ground
    d = p.d
    ψ = integrator.u
    
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
    
    p_norm = p⁺ + p⁰ + p⁻
    rn = rand() * p_norm
    for i ∈ 1:n_ground
        ψ[i] = 0.0
    end
    
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
    
    # is this required?
    ψ_norm = 0.0
    for i ∈ 1:n_states
        ψ_norm += norm(ψ[i])^2
    end
    ψ_norm = sqrt(ψ_norm)
    for i ∈ 1:n_states
        ψ[i] /= ψ_norm
    end
    
    if count_scatter
        p.n_scatters += 1
    end
    
    # zero excited state populations
    for i ∈ (n_states+1):(n_states+n_excited)
        integrator.u[i] = 0.0
    end

    # # add a repumper photon kick along the x-axis
    # repump_states_prob = 1/150 # 1/150 photons are repumped with a laser that is not retroreflected
    # if rand() < repump_states_prob
    #     integrator.u[n_states + n_excited + 4] += 2 / p.mass # 2 photons are required to repump on average
    # end
    
    # add two photon recoils
    for _ ∈ 1:number_recoils
        dp = sample_direction(1)
        dv = dp ./ p.mass
        integrator.u[n_states + n_excited + 4] += dv[1]
        integrator.u[n_states + n_excited + 5] += dv[2]
        integrator.u[n_states + n_excited + 6] += dv[3]
    end
    
    p.time_to_decay = rand(p.decay_dist)

    return nothing
end
export SE_collapse_pol_always_repump!

function spontaneous_emission_event(p, i_excited)
    """ 
        Excited state i_excited sponatneously emits. Randomly sample which ground state it decays into, 
        return the ground state index and change in m_F (which is relevant in determining the direction of
        momentum kick).
        
    """
    n_states = length(p.states)
    
    transition_probs = norm.(p.d[:,i_excited,:]).^2
    w = weights(transition_probs)
    i = sample(w)
    δm = -((i-1)÷n_states - 2)
    i_ground = (i-1) % n_states + 1
    # @printf("decay from %i to %i", i_excited, i_ground)
    # println()

    return (i_ground, δm)
end

uniform_dist = Uniform(0, 2π)
function sample_direction(r=1.0)
    θ = 2π * rand()
    z = rand() * 2 - 1
    return (r * sqrt(1 - z^2) * cos(θ), r * sqrt(1 - z^2) * sin(θ), r * z)
end
export sample_direction

function schrodinger_stochastic(
    particle, states, fields, d, ψ₀, mass, n_excited;
    sim_params=nothing, extra_data=nothing, λ=1.0, Γ=2π, update_H_and_∇H=update_H_and_∇H, diffusion_constant=0.0)
    """
    extra_p should contain n_excited
    
    ψ in the output will be of the following format:
    the first n_states indicies will be the coefficients of the current state;
    the next n_excited indicies is the time-integrated excited state population (reset by callbacks);
    the next 3 indicies are the current position;
    the next 3 indicies are the current velocity;
    the last 3 indicies are the current force.
    """

    n_states = length(states)
    n_fields = length(fields)

    states = StructArray(states)
    fields = StructArray(fields)

    k = 2π / λ
    
    # time unit: 1/Γ
    for i ∈ eachindex(fields)
        fields.ω[i] /= Γ
    end
    for i ∈ eachindex(states)
        states.E[i] *= 2π
        states.E[i] /= Γ
    end

    r0 = particle.r0
    r = particle.r
    v = particle.v

    type_complex = ComplexF64

    H = StructArray( zeros(type_complex, n_states, n_states) )
    H₀ = deepcopy(H)
    ∇H = SVector{3, ComplexF64}(0,0,0)

    ω = [s.E for s in states]
    eiωt = StructArray(zeros(type_complex, n_states))

    # Compute cartesian indices to indicate nonzero transition dipole moments in `d`
    # Indices below the diagonal of the Hamiltonian are removed, since those are defined via taking the conjugate
    d_nnz_m = [cart_idx for cart_idx ∈ findall(d[:,:,1] .!= 0) if cart_idx[2] >= cart_idx[1]]
    d_nnz_0 = [cart_idx for cart_idx ∈ findall(d[:,:,2] .!= 0) if cart_idx[2] >= cart_idx[1]]
    d_nnz_p = [cart_idx for cart_idx ∈ findall(d[:,:,3] .!= 0) if cart_idx[2] >= cart_idx[1]]
    d_nnz = [d_nnz_m, d_nnz_0, d_nnz_p]

    Js = Array{Jump}(undef, 0)
    ds = [Complex{Float64}[], Complex{Float64}[], Complex{Float64}[]]
    ds_state1 = [Int64[], Int64[], Int64[]]
    ds_state2 = [Int64[], Int64[], Int64[]]
    for s′ in eachindex(states), s in s′:n_states, q in qs
        dme = d[s′, s, q+2]
        if abs(dme) > 1e-10 && (states[s′].E < states[s].E) # only energy-allowed jumps are generated
        # if (states[s′].E < states[s].E) # only energy-allowed jumps are generated
            push!(ds_state1[q+2], s)
            push!(ds_state2[q+2], s′)
            push!(ds[q+2], dme)
            rate = norm(dme)^2 / 2
            J = Jump(s, s′, q, rate)
            push!(Js, J)
        end
    end
    ds = [StructArray(ds[1]), StructArray(ds[2]), StructArray(ds[3])]
    
    ψ_soa = StructArray(ψ₀)
    dψ_soa = StructArray(ψ₀)
    
    # ψ contains the state vector, accumulated excited state populations, position, velocity, force
    ψ = zeros(ComplexF64, n_states + n_excited + 3 + 3 + 3)
    ψ[1:n_states] .= ψ₀
    ψ[n_states + n_excited + 1: n_states + n_excited + 3] .= r
    ψ[n_states + n_excited + 4: n_states + n_excited + 6] .= v
    dψ = deepcopy(ψ)

    E = @SVector Complex{Float64}[0,0,0]
    E_k = [@SVector Complex{Float64}[0,0,0] for _ ∈ 1:3]

    decay_dist = Exponential(1)

    # NOTE: mass with correct unit = dimensionless mass here * hbar * k^2 / Γ
    p = MutableNamedTuple(
        H=H, H₀=H₀, ∇H=∇H, ψ=ψ, dψ=dψ, ψ_soa=ψ_soa, dψ_soa=dψ_soa, ω=ω, eiωt=eiωt, Js=Js,
        states=states, fields=fields, r0=r0, r=r, v=v, d=d, d_nnz=d_nnz, 
        λ=λ, k=k, Γ=Γ,
        E=E, E_k=E_k,
        ds=ds, ds_state1=ds_state1, ds_state2=ds_state2,
        sim_params=sim_params, extra_data=extra_data, mass = mass, update_H_and_∇H=update_H_and_∇H, populations = zeros(Float64, n_states),
        n_scatters = 0,
        save_counter=0,
        n_states=length(states),
        n_ground=length(states) - n_excited,
        n_excited=n_excited,
        trajectory=Vector{ComplexF64}[],
        decay_dist=decay_dist,
        time_to_decay=rand(decay_dist),
        last_decay_time=0.0,
        diffusion_constant=diffusion_constant,
        time_before_decay=0.0
        )

    return p
end
export schrodinger_stochastic

# Condition function used to determine when the particle decays
# function condition(u,t,integrator)
#     p = integrator.p
#     integrated_excited_pop = 0.0
#     for i ∈ 1:p.n_excited
#         integrated_excited_pop += real(u[p.n_states+i])
#     end
#     _condition = integrated_excited_pop - p.time_to_decay
#     return _condition
# end
# export condition

function condition_discrete(u,t,integrator)
    p = integrator.p
    integrated_excited_pop = zero(Float64)
    @inbounds @fastmath for i ∈ 1:p.n_excited
        integrated_excited_pop += real(u[p.n_states+i])
    end
    _condition = integrated_excited_pop - p.time_to_decay
    return _condition > 0
end
export condition_discrete

function ψ_stochastic!(dψ, ψ, p, τ)
    @unpack ψ_soa, dψ_soa, r, H₀, ω, fields, H, E_k, ds, ds_state1, ds_state2, Js, eiωt, states, extra_p, mass = p
    
    n_states = length(states)
    n_excited = p.n_excited
    
    r = SVector(real(ψ[n_states + n_excited + 1]), real(ψ[n_states + n_excited + 2]), real(ψ[n_states + n_excited + 3]))

    ψ_norm = 0.0
    for i ∈ 1:n_states
        ψ_norm += norm(ψ[i])^2
    end
    ψ_norm = sqrt(ψ_norm)
    for i ∈ 1:n_states
        ψ[i] /= ψ_norm
    end
    
    base_to_soa!(ψ, ψ_soa)
    
    update_H!(p, τ, r, H₀, fields, H, E_k, ds, ds_state1, ds_state2, Js)
    
    update_eiωt!(eiωt, ω, τ)
    Heisenberg!(H, eiωt)

    mul_by_im_minus!(ψ_soa)
    mul_turbo!(dψ_soa, H, ψ_soa)
    
    soa_to_base!(dψ, dψ_soa)
    
    # calculate force
    f = force_stochastic(E_k, ds, ds_state1, ds_state2, ψ_soa, eiωt)

    # accumulate excited state populations
    for i ∈ 1:n_excited
        dψ[n_states + i] = norm(ψ[n_states - n_excited + i])^2
    end
    
    for i ∈ 1:3
        dψ[n_states + n_excited + i] = ψ[n_states + n_excited + i + 3] # update position
        dψ[n_states + n_excited + 3 + i] = f[i]/mass # update velocity
    end

    # update force
    ψ[(n_states + n_excited + 6 + 1):(n_states + n_excited + 6 + 3)] .= f
    dψ[(n_states + n_excited + 6 + 1):(n_states + n_excited + 6 + 3)] .= 0
    
    return nothing
end
export ψ_stochastic!

# function force_stochastic(E_k, ds, ds_state1, ds_state2, ψ_soa, eiωt)
#     F = @SVector Complex{Float64}[0,0,0]

#     @inbounds @fastmath for q ∈ 1:3
#         ds_q = ds[q]
#         ds_q_re = ds_q.re
#         ds_q_im = ds_q.im
#         ds_state1_q = ds_state1[q]
#         ds_state2_q = ds_state2[q]
#         for k ∈ 1:3
#             E_kq = E_k[k][q]
#             E_kq_re = real(E_kq)
#             E_kq_im = imag(E_kq)
#             F_k_re = 0.0
#             F_k_im = 0.0
#             for j ∈ eachindex(ds_q)
#                 m = ds_state1_q[j] # excited state
#                 n = ds_state2_q[j] # ground state
                
#                 # construct ρ_mn = c_m c_n^*
#                 # ρ_mn = conj(ψ_soa[n]*eiωt[n]) * ψ_soa[m]*eiωt[m]

#                 c_m = ψ_soa[m] * conj(eiωt[m]) # exp(-iωt) factor to transform to Heisenberg picture
#                 c_n = ψ_soa[n] * conj(eiωt[n]) # exp(-iωt) factor to transform to Heisenberg picture

#                 ρ_mn = c_m * conj(c_n)

#                 ρ_re = real(ρ_mn)
#                 ρ_im = imag(ρ_mn)
                
#                 d_re = ds_q_re[j]
#                 d_im = ds_q_im[j]

#                 a1 = d_re * ρ_re - d_im * ρ_im
#                 a2 = d_re * ρ_im + d_im * ρ_re
#                 F_k_re += E_kq_re * a1 - E_kq_im * a2
#                 F_k_im += E_kq_im * a1 + E_kq_re * a2     
#             end
#             F -= (im * F_k_re - F_k_im) * ê[k] # multiply by im
#         end
#     end
#     F += conj(F)

#     return real.(F)
# end

function force_stochastic_v2(E_k, ds, ds_state1, ds_state2, ψ_soa, eiωt)
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
                d_re = ds_q_re[j]
                d_im = ds_q_im[j]

                c_m_re = ψ_soa.re[m]
                c_m_im = ψ_soa.im[m]
                c_n_re = ψ_soa.re[n]
                c_n_im = -ψ_soa.im[n] # taking the conjugate 
                
                ρ_re = c_m_re * c_n_re - c_m_im * c_n_im
                ρ_im = c_m_re * c_n_im + c_m_im * c_n_re

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

function force_stochastic_nocomplex(n_states, E_k_re, E_k_im, ds, ds_state1, ds_state2, ψ, eiωt_re, eiωt_im)
    F = @SVector Complex{Float64}[0,0,0]

    @inbounds @fastmath for q ∈ 1:3
        ds_q = ds[q]
        ds_q_re = ds_q.re
        ds_q_im = ds_q.im
        ds_state1_q = ds_state1[q]
        ds_state2_q = ds_state2[q]
        for k ∈ 1:3
            E_kq_re = E_k_re[k][q]
            E_kq_im = E_k_im[k][q]
            F_k_re = 0.0
            F_k_im = 0.0
            for j ∈ eachindex(ds_q)
                m = ds_state1_q[j] # excited state
                n = ds_state2_q[j] # ground state
                
                # construct ρ_mn = c_m c_n^*

                c_m = (ψ[m] + im * ψ[n_states+m]) * conj(eiωt_re[m] + im * eiωt_im[m]) # exp(-iωt) factor to transform to Heisenberg picture
                c_n = (ψ[n] + im * ψ[n_states+n]) * conj(eiωt_re[n] + im * eiωt_im[n]) # exp(-iωt) factor to transform to Heisenberg picture

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

function force_stochastic_nocomplex_v0(n_states, E_k, ds, ds_state1, ds_state2, ψ, eiωt)
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

                c_m = (ψ[m] + im * ψ[n_states+m]) * conj(eiωt[m]) # exp(-iωt) factor to transform to Heisenberg picture
                c_n = (ψ[n] + im * ψ[n_states+n]) * conj(eiωt[n]) # exp(-iωt) factor to transform to Heisenberg picture

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

function ψ_stochastic_nocomplex_v0!(dψ, ψ, p, τ)

    @unpack ω, fields, H, H₀, E_k, ds, ds_state1, ds_state2, Js, eiωt, states, extra_p, mass = p
    
    n_states = length(states)
    n_excited = p.n_excited
    
    r = SVector(real(ψ[2n_states + n_excited + 1]), real(ψ[2n_states + n_excited + 2]), real(ψ[2n_states + n_excited + 3]))

    ψ_norm = 0.0
    for i ∈ 1:n_states
        ψ_norm += ψ[i]^2 + ψ[n_states+i]^2
    end
    ψ_norm = sqrt(ψ_norm)
    for i ∈ 1:2n_states
        ψ[i] /= ψ_norm
    end

    update_H!(p, τ, r, H₀, fields, H, E_k, ds, ds_state1, ds_state2, Js)
    
    update_eiωt!(eiωt, ω, τ)
    Heisenberg!(H, eiωt)

    # multiply ψ by -im
    @turbo for i ∈ 1:n_states
        ψ_re = ψ[i]
        ψ[i] = ψ[n_states+i]
        ψ[n_states+i] = -ψ_re
    end
    
    # mul_turbo_nocomplex_v0!(n_states, dψ, H, ψ)
    mul_turbo_nocomplex_v00!(n_states, dψ, H, ψ)
    
    # calculate force
    f = force_stochastic_nocomplex_v0(n_states, E_k, ds, ds_state1, ds_state2, ψ, eiωt)

    # accumulate excited state populations
    for i ∈ 1:n_excited
        dψ[2n_states + i] = ψ[n_states-n_excited+i]^2 + ψ[2n_states-n_excited+i]^2
        # println(ψ[n_states-n_excited+i]^2 + ψ[2n_states-n_excited+i]^2)
    end
    
    for i ∈ 1:3
        dψ[2n_states + n_excited + i] = ψ[2n_states + n_excited + i + 3] # update position
        dψ[2n_states + n_excited + i + 3] = f[i]/mass # update velocity
    end

    ψ[end-2:end] .= f
    dψ[end-2:end] .= 0
    
    return nothing
end
export ψ_stochastic_nocomplex_v0!

function schrodinger_stochastic_nocomplex_v0(
    particle, states, fields, d, ψ₀, mass, n_excited;
    extra_p=nothing, λ=1.0, Γ=2π, update_H=update_H)

    n_states = length(states)

    states = StructArray(states)
    fields = StructArray(fields)

    k = 2π / λ
    
    # time unit: 1/Γ
    for i ∈ eachindex(fields)
        fields.ω[i] /= Γ
    end
    for i ∈ eachindex(states)
        states.E[i] *= 2π
        states.E[i] /= Γ
    end

    r0 = particle.r0
    r = particle.r
    v = particle.v

    type_complex = Complex{Float64}

    H = StructArray( zeros(type_complex, n_states, n_states) )
    H₀ = deepcopy(H)

    ω = [s.E for s in states]
    eiωt = StructArray(zeros(type_complex, n_states))

    # Compute cartesian indices to indicate nonzero transition dipole moments in `d`
    # Indices below the diagonal of the Hamiltonian are removed, since those are defined via taking the conjugate
    d_nnz_m = [cart_idx for cart_idx ∈ findall(d[:,:,1] .!= 0) if cart_idx[2] >= cart_idx[1]]
    d_nnz_0 = [cart_idx for cart_idx ∈ findall(d[:,:,2] .!= 0) if cart_idx[2] >= cart_idx[1]]
    d_nnz_p = [cart_idx for cart_idx ∈ findall(d[:,:,3] .!= 0) if cart_idx[2] >= cart_idx[1]]
    d_nnz = [d_nnz_m, d_nnz_0, d_nnz_p]

    # Define jumps
    Js = Array{Jump}(undef, 0)
    ds = [Complex{Float64}[], Complex{Float64}[], Complex{Float64}[]]
    ds_state1 = [Int64[], Int64[], Int64[]]
    ds_state2 = [Int64[], Int64[], Int64[]]

    for s′ in eachindex(states), s in s′:n_states, q in qs
        dme = d[s′, s, q+2]
        if abs(dme) > 1e-10 && (states[s′].E < states[s].E) # only energy-allowed jumps are generated
        # if (states[s′].E < states[s].E) # only energy-allowed jumps are generated
            push!(ds_state1[q+2], s)
            push!(ds_state2[q+2], s′)
            push!(ds[q+2], dme)
            rate = norm(dme)^2 / 2
            J = Jump(s, s′, q, rate)
            push!(Js, J)
        end
    end
    ds = [StructArray(ds[1]), StructArray(ds[2]), StructArray(ds[3])]
    
    # ψ contains the state vector (real and imaginary separated), accumulated excited state populations, position, velocity.
    ψ = zeros(Float64, 2n_states + n_excited + 6 + 3)

    ψ[1:n_states] .= ψ₀
    ψ[2n_states + n_excited + 1:2n_states + n_excited + 3] .= r
    ψ[2n_states + n_excited + 4:2n_states + n_excited + 6] .= v

    E = @SVector Complex{Float64}[0,0,0]
    E_k = [@SVector Complex{Float64}[0,0,0] for _ ∈ 1:3]

    decay_dist = Exponential(1)

    # NOTE: mass with correct unit = dimensionless mass here * hbar * k^2 / Γ
    p = MutableNamedTuple(
        ψ=ψ,
        H=H,
        H₀=H₀,
        ω=ω, 
        eiωt=eiωt,
        Js=Js,
        states=states, fields=fields, r0=r0, r=r, v=v, d=d, d_nnz=d_nnz, λ=λ,
        k=k,
        E=E,
        E_k=E_k,
        ds=ds, ds_state1=ds_state1, ds_state2=ds_state2,
        extra_p=extra_p, mass = mass, update_H = update_H, populations = zeros(Float64, n_states),
        n_scatters = 0,
        save_counter=0,
        n_states=length(states),
        n_ground=length(states) - n_excited,
        n_excited=n_excited,
        trajectory=Vector{Float64}[],
        decay_dist=decay_dist,
        time_to_decay=rand(decay_dist)
        )

    return p
end
export schrodinger_stochastic_nocomplex_v0

# p should have:
# a "dark state" lifetime distribution extra_p.dark_time_dist
# p.dark_time, save sampled value from the distribution
# a boolean variable p.is_dark
# p.dark_t0
# p.FC_mainline

function schrodinger_stochastic_repump(
    particle, states, fields, d, ψ₀, mass, n_excited;
    sim_params=nothing, extra_data=nothing, λ=1.0, Γ=2π, update_H_and_∇H=update_H_and_∇H, dark_lifetime=0.0, FC_mainline=1.0)
    """
    dark_lifetime = time the molecule spends in a dark (to mainline laser) state
    FC_mainline = Frank-Condon factor of mainline transtion
    
    extra_p should contain n_excited
    
    ψ in the output will be of the following format:
    the first n_states indicies will be the coefficients of the current state;
    the next n_excited indicies is the time-integrated excited state population (reset by callbacks);
    the next 3 indicies are the current position;
    the next 3 indicies are the current velocity;
    the last 3 indicies are the current force.
    """

    n_states = length(states)
    n_fields = length(fields)

    states = StructArray(states)
    fields = StructArray(fields)

    k = 2π / λ
    
    # time unit: 1/Γ
    for i ∈ eachindex(fields)
        fields.ω[i] /= Γ
    end
    for i ∈ eachindex(states)
        states.E[i] *= 2π
        states.E[i] /= Γ
    end

    r0 = particle.r0
    r = particle.r
    v = particle.v

    type_complex = ComplexF64

    H = StructArray( zeros(type_complex, n_states, n_states) )
    H₀ = deepcopy(H)
    ∇H = SVector{3, ComplexF64}(0,0,0)

    ω = [s.E for s in states]
    eiωt = StructArray(zeros(type_complex, n_states))

    # Compute cartesian indices to indicate nonzero transition dipole moments in `d`
    # Indices below the diagonal of the Hamiltonian are removed, since those are defined via taking the conjugate
    d_nnz_m = [cart_idx for cart_idx ∈ findall(d[:,:,1] .!= 0) if cart_idx[2] >= cart_idx[1]]
    d_nnz_0 = [cart_idx for cart_idx ∈ findall(d[:,:,2] .!= 0) if cart_idx[2] >= cart_idx[1]]
    d_nnz_p = [cart_idx for cart_idx ∈ findall(d[:,:,3] .!= 0) if cart_idx[2] >= cart_idx[1]]
    d_nnz = [d_nnz_m, d_nnz_0, d_nnz_p]

    Js = Array{Jump}(undef, 0)
    ds = [Complex{Float64}[], Complex{Float64}[], Complex{Float64}[]]
    ds_state1 = [Int64[], Int64[], Int64[]]
    ds_state2 = [Int64[], Int64[], Int64[]]
    for s′ in eachindex(states), s in s′:n_states, q in qs
        dme = d[s′, s, q+2]
        if abs(dme) > 1e-10 && (states[s′].E < states[s].E) # only energy-allowed jumps are generated
        # if (states[s′].E < states[s].E) # only energy-allowed jumps are generated
            push!(ds_state1[q+2], s)
            push!(ds_state2[q+2], s′)
            push!(ds[q+2], dme)
            rate = norm(dme)^2 / 2
            J = Jump(s, s′, q, rate)
            push!(Js, J)
        end
    end
    ds = [StructArray(ds[1]), StructArray(ds[2]), StructArray(ds[3])]
    
    ψ_soa = StructArray(ψ₀)
    dψ_soa = StructArray(ψ₀)
    
    # ψ contains the state vector, accumulated excited state populations, position, velocity, force
    ψ = zeros(ComplexF64, n_states + n_excited + 3 + 3 + 3)
    ψ[1:n_states] .= ψ₀
    ψ[n_states + n_excited + 1: n_states + n_excited + 3] .= r
    ψ[n_states + n_excited + 4: n_states + n_excited + 6] .= v
    dψ = deepcopy(ψ)

    E = @SVector Complex{Float64}[0,0,0]
    E_k = [@SVector Complex{Float64}[0,0,0] for _ ∈ 1:3]

    decay_dist = Exponential(1)
    dark_time_dist = Exponential(dark_lifetime)

    # NOTE: mass with correct unit = dimensionless mass here * hbar * k^2 / Γ
    p = MutableNamedTuple(
        H=H, H₀=H₀, ∇H=∇H, ψ=ψ, dψ=dψ, ψ_soa=ψ_soa, dψ_soa=dψ_soa, ω=ω, eiωt=eiωt, Js=Js,
        states=states, fields=fields, r0=r0, r=r, v=v, d=d, d_nnz=d_nnz, 
        λ=λ, k=k, Γ=Γ,
        E=E, E_k=E_k,
        ds=ds, ds_state1=ds_state1, ds_state2=ds_state2,
        sim_params=sim_params, extra_data=extra_data, mass = mass, update_H_and_∇H=update_H_and_∇H, populations = zeros(Float64, n_states),
        n_scatters = 0,
        save_counter=0,
        n_states=length(states),
        n_ground=length(states) - n_excited,
        n_excited=n_excited,
        trajectory=Vector{ComplexF64}[],
        decay_dist=decay_dist,
        time_to_decay=rand(decay_dist),
        dark_time_dist = dark_time_dist,
        dark_time = rand(dark_time_dist),
        dark_time_t0 = 0.0,
        is_dark = false,
        FC_mainline = FC_mainline
        )

    return p
end
export schrodinger_stochastic_repump

function ψ_stochastic_repump!(dψ, ψ, p, t)
    if ~p.is_dark
        ψ_stochastic_potential!(dψ, ψ, p, t)
    else
        for i ∈ 1:(p.n_states + p.n_excited)
            dψ[i] = 0.0
        end
        for i ∈ (p.n_states + p.n_excited + 4):(p.n_states + p.n_excited + 6)
            dψ[i] = 0.0
        end
    end
    return nothing
end
export ψ_stochastic_repump!

# function ψ_stochastic_repump!(dψ, ψ, p, τ)
      
#     if p.is_dark == false
#         ψ_stochastic_potential!(dψ, ψ, p, τ)
#         return nothing
#     else
#        @unpack ψ_soa, dψ_soa, r, H₀, ω, fields, H, E_k, ds, ds_state1, ds_state2, Js, eiωt, states, extra_p, mass = p
        
#         # n_states = length(states)
#         # n_excited = p.n_excited

#         # r = SVector(real(ψ[n_states + n_excited + 1]), real(ψ[n_states + n_excited + 2]), real(ψ[n_states + n_excited + 3]))
        
#         # for i ∈ 1:(n_excited + n_states)
#         #     dψ[i] = 0.0
#         # end

#         return nothing 
#     end
# end

function SE_collapse_repump!(integrator)
    # if the collapse condition has been satisfied and the molecule was in a dark state, move it back to a bright state (all states equally likely)
    if integrator.p.is_dark

        integrator.p.is_dark = false
        n_states = length(integrator.p.states)
        n_excited = integrator.p.n_excited

        # excite molecule to a random excited state
        i = Int(floor(rand()*n_excited)) + 1
        for i ∈ 1:n_states
            integrator.u[i] = 0.0
        end
        integrator.u[n_states - n_excited + i] = 1.0

        # decay back down to the ground state with a single photon recoil
        SE_collapse_pol_always_repump!(integrator, false, 1)
        
    else
        # scatter; add two photon scatters for diffusion from decay AND absorption
        SE_collapse_pol_always_repump!(integrator, true, 2)

        # decay to a dark state
        if rand() > integrator.p.FC_mainline
            integrator.p.is_dark = true
            integrator.p.dark_time = rand(integrator.p.dark_time_dist)
            integrator.p.dark_time_t0 = integrator.t
        end
    end
end
export SE_collapse_repump!

# Condition function used to determine when the particle decays
function condition(u,t,integrator)
    p = integrator.p
    k = 2π/p.λ
    integrated_excited_pop = 0.0
    for i ∈ 1:p.n_excited
        integrated_excited_pop += real(u[p.n_states+i])
    end
    
    _condition = integrated_excited_pop - p.time_to_decay
    
    r = 0.0
    for i ∈ 1:3
        r += norm(u[p.n_states + p.n_excited + i])^2
    end
    r = sqrt(r)
    if r >= 2e-3*k # terminate if the particle is more than 2 mm from the centre
       terminate!(integrator)
    elseif integrator.p.n_scatters > integrator.p.sim_params.photon_budget # also terminate if too many photons have been scattered
        terminate!(integrator)
    end
    return _condition
end
export condition

function condition_simple(u,t,integrator)
    p = integrator.p
    k = 2π/p.λ
    integrated_excited_pop = 0.0
    for i ∈ 1:p.n_excited
        integrated_excited_pop += real(u[p.n_states+i])
    end
    
    _condition = integrated_excited_pop - p.time_to_decay
    
    r = 0.0
    for i ∈ 1:3
        r += norm(u[p.n_states + p.n_excited + i])^2
    end
    r = sqrt(r)
    # condition for capture
    if real(u[p.n_states + p.n_excited + 1] + u[p.n_states + p.n_excited + 2]) / √2 >= 25e-3*k # terminate if the particle is more than 20 mm from the centre
       terminate!(integrator)
    end
    return _condition
end
export condition_simple

function update_H_and_∇H(H, p, r, t)
    k = 2π / p.λ
    
    # Define a ramping magnetic field
    Zeeman_Hz = p.extra_data.Zeeman_Hz
    Zeeman_Hx = p.extra_data.Zeeman_Hx
    Zeeman_Hy = p.extra_data.Zeeman_Hy
    
    τ_bfield = p.sim_params.B_ramp_time 
    scalar = t/τ_bfield
    scalar = min(scalar, 1.0)
    
    gradient_x = +scalar * p.sim_params.B_gradient * 1e2 / k/2
    gradient_y = +scalar * p.sim_params.B_gradient * 1e2 / k/2
    gradient_z = -scalar * p.sim_params.B_gradient * 1e2 / k
    
    Bx = gradient_x * r[1] + p.sim_params.B_offset[1]
    By = gradient_y * r[2] + p.sim_params.B_offset[2]
    Bz = gradient_z * r[3] + p.sim_params.B_offset[3]
    
    @turbo for i in eachindex(H)
        H.re[i] = Bz * Zeeman_Hz.re[i] + Bx * Zeeman_Hx.re[i] + By * Zeeman_Hy.re[i]
        H.im[i] = Bz * Zeeman_Hz.im[i] + Bx * Zeeman_Hx.im[i] + By * Zeeman_Hy.im[i]
    end
    
    # # Update the Hamiltonian for the molecule-ODT interaction
    # H_ODT = p.extra_data.H_ODT_static
    
    # ODT_size = p.sim_params.ODT_size .* p.k
    # update_ODT_center!(p, t)
    # ODT_x = p.extra_data.ODT_position[1] * p.k
    # ODT_z = p.extra_data.ODT_position[2] * p.k
    
    # scalar_ODT = exp(-2(r[1]-ODT_x)^2/ODT_size[1]^2) * exp(-2r[2]^2/ODT_size[2]^2) * exp(-2(r[3]-ODT_z)^2/ODT_size[3]^2)
    
    # @turbo for i in eachindex(H)
    #     H.re[i] += H_ODT.re[i] * scalar_ODT
    #     H.im[i] += H_ODT.im[i] * scalar_ODT
    # end
    
    return SVector{3,ComplexF64}(0,0,0)
    # return SVector{3,ComplexF64}((-4(r[1]-ODT_x) / ODT_size[1]^2) * scalar_ODT, (-4r[2] / ODT_size[2]^2) * scalar_ODT, (-4(r[3]-ODT_z) / ODT_size[3]^2) * scalar_ODT)
    
end


function condition_repump(u, t, integrator)
    p = integrator.p
    _condition = 0.0
    if integrator.p.is_dark
        _condition += (t - p.dark_time_t0) - p.dark_time
    else
        integrated_excited_pop = 0.0
        @inbounds @fastmath for i ∈ 1:p.n_excited
            integrated_excited_pop += real(u[p.n_states+i])
        end
        _condition += integrated_excited_pop - p.time_to_decay
    end

    # terminate if particle is too far from center
    # r = 0.0
    # for i ∈ 1:3
    #     r += norm(u[p.n_states + p.n_excited + i])^2
    # end
    # r = sqrt(r)
    # if r >= 2e-3 * integrator.p.k
    #    terminate!(integrator)
    # end

    return _condition
end
export condition_repump

function ψ_stochastic_potential!(dψ, ψ, p, t)
    @unpack ψ_soa, dψ_soa, r, ω, fields, H, H₀, ∇H, E_k, ds, ds_state1, ds_state2, Js, eiωt, states, extra_data, mass, k, Γ = p
    
    n_states = length(states)
    n_excited = p.n_excited
    
    r = SVector(real(ψ[n_states + n_excited + 1]), real(ψ[n_states + n_excited + 2]), real(ψ[n_states + n_excited + 3]))

    ψ_norm = 0.0
    for i ∈ 1:n_states
        ψ_norm += norm(ψ[i])^2
    end
    ψ_norm = sqrt(ψ_norm)
    for i ∈ 1:n_states
        ψ[i] /= ψ_norm
    end
    
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

    # add gravity to the force
    g = -9.81 / (Γ^2/k)
    f += SVector{3,Float64}(0,mass*g,0)

    # accumulate excited state populations
    for i ∈ 1:n_excited
        dψ[n_states + i] = norm(ψ[n_states - n_excited + i])^2
    end
    
    for i ∈ 1:3
        dψ[n_states + n_excited + i] = ψ[n_states + n_excited + i + 3] # update position
        dψ[n_states + n_excited + 3 + i] = f[i] / mass # update velocity
    end

    # update force
    # ψ[(n_states + n_excited + 6 + 1):(n_states + n_excited + 6 + 3)] .= f/ψ_norm
    # dψ[(n_states + n_excited + 6 + 1):(n_states + n_excited + 6 + 3)] .= 0

    # diffusion_direction = 1.0
    # if rand() < 0.5
    #     diffusion_direction= -1
    # end
    # dψ[n_states + n_excited + 4] += p.diffusion_constant * diffusion_direction / (p.mass * sqrt(dt))
    
    # diffusion_direction = 1.0
    # if rand() < 0.5
    #     diffusion_direction= -1
    # end
    # dψ[n_states + n_excited + 5] += p.diffusion_constant * diffusion_direction / (p.mass * sqrt(dt))
    
    # diffusion_direction = 1.0
    # if rand() < 0.5
    #     diffusion_direction= -1
    # end
    # dψ[n_states + n_excited + 6] += p.diffusion_constant * diffusion_direction / (p.mass * sqrt(dt))

    return nothing
end
export ψ_stochastic_potential!

function ψ_stochastic_potential_v2!(dψ, ψ, p, t)
    # (1) transforms ψ rather than H 

    @unpack ψ_soa, dψ_soa, r, ω, fields, H, H₀, ∇H, E_k, ds, ds_state1, ds_state2, Js, eiωt, states, extra_data, mass, k, Γ = p
    
    n_states = length(states)
    n_excited = p.n_excited
    
    r = SVector(real(ψ[n_states + n_excited + 1]), real(ψ[n_states + n_excited + 2]), real(ψ[n_states + n_excited + 3]))

    ψ_norm = 0.0
    for i ∈ 1:n_states
        ψ_norm += norm(ψ[i])^2
    end
    ψ_norm = sqrt(ψ_norm)
    for i ∈ 1:n_states
        ψ[i] /= ψ_norm
    end
    
    update_eiωt!(eiωt, ω, t)
    base_to_soa!(ψ, ψ_soa)
    Heisenberg_turbo_state!(ψ_soa, eiωt, -1)

    update_H!(p, t, r, fields, H, E_k, ds, ds_state1, ds_state2, Js) # molecule-light Hamiltonian in schrodinger picutre
    ∇H = p.update_H_and_∇H(H₀, p, r, t) # Zeeman and ODT hamiltonian in schrodinger picutre

    @turbo for i ∈ eachindex(H)
        H.re[i] += H₀.re[i]
        H.im[i] += H₀.im[i]
    end
    
    mul_by_im_minus!(ψ_soa)
    mul_turbo!(dψ_soa, H, ψ_soa)
    
    Heisenberg_turbo_state!(dψ_soa, eiωt)
    soa_to_base!(dψ, dψ_soa)

    # force calculations
    f = 0*force_stochastic_v2(E_k, ds, ds_state1, ds_state2, ψ_soa, eiωt) # force due to lasers

    H₀_expectation = operator_matrix_expectation(H₀, ψ_soa)
    f += ∇H .* (-H₀_expectation) # force due to conservative potential

    # add gravity to the force
    # g = -9.81 / (Γ^2/k)
    # f += SVector{3,Float64}(0,mass*g,0)

    # # accumulate excited state populations
    # for i ∈ 1:n_excited
    #     dψ[n_states + i] = norm(ψ[n_states - n_excited + i])^2
    # end
    
    for i ∈ 1:3
        dψ[n_states + n_excited + i] = ψ[n_states + n_excited + i + 3] # update position
        dψ[n_states + n_excited + 3 + i] = f[i]/mass # update velocity
    end
    
    return nothing
end
export ψ_stochastic_potential_v2!

function ψ_stochastic_oop!(ψ, p, t)
    @unpack dψ, ψ_soa, dψ_soa, r, ω, fields, H, H₀, ∇H, E_k, ds, ds_state1, ds_state2, Js, eiωt, states, extra_data, mass, k, Γ = p
    
    n_states = length(states)
    n_excited = p.n_excited
    
    r = SVector(real(ψ[n_states + n_excited + 1]), real(ψ[n_states + n_excited + 2]), real(ψ[n_states + n_excited + 3]))

    ψ_norm = 0.0
    for i ∈ 1:n_states
        ψ_norm += norm(ψ[i])^2
    end
    ψ_norm = sqrt(ψ_norm)
    for i ∈ 1:n_states
        ψ[i] /= ψ_norm
    end
    
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

    # add gravity to the force
    g = -9.81 / (Γ^2/k)
    f += SVector{3,Float64}(0,mass*g,0)

    # accumulate excited state populations
    for i ∈ 1:n_excited
        dψ[n_states + i] = norm(ψ[n_states - n_excited + i])^2
    end
    
    for i ∈ 1:3
        dψ[n_states + n_excited + i] = ψ[n_states + n_excited + i + 3] # update position
        dψ[n_states + n_excited + 3 + i] = f[i]/mass # update velocity
    end

    return SVector{length(dψ), ComplexF64}(dψ)
end
export ψ_stochastic_oop!

""" 
    Compute the expectation value ⟨ψ|O|ψ⟩ for an operator O and a state vector |ψ⟩. This function assumes that O is real.
"""
function operator_matrix_expectation(O_Heisenberg, state)
    O_exp = zero(Float64)
    @turbo for i ∈ eachindex(state)
        state_i_re = state.re[i]
        state_i_im = state.im[i]
        for j ∈ eachindex(state)
            # second term in addition below is the imaginary part, which has a positive sign because we take the conjugate of state
            O_exp += O_Heisenberg.re[i,j] * (state_i_re * state.re[j] + state_i_im * state.im[j])
        end
    end
    return O_exp
end
export operator_matrix_expectation

function operator_matrix_expectation_column(O_Heisenberg, state)
    O_exp = zero(Float64)
    @turbo for j ∈ eachindex(state)
        state_j_re = state.re[j]
        state_j_im = state.im[j]
        for i ∈ eachindex(state)
            # second term in addition below is the imaginary part, which has a positive sign because we take the conjugate of state
            O_exp += O_Heisenberg.re[i,j] * (state_j_re * state.re[i] + state_j_im * state.im[i])
        end
    end
    return O_exp
end
export operator_matrix_expectation_column

function extend_operator(operator::T, state, state′, args...) where {T}
    val = zero(ComplexF64)
    for (i, basis_state) in enumerate(state.basis)
        for (j, basis_state′) in enumerate(state′.basis)
            val += conj(state.coeffs[i]) * state′.coeffs[j] * operator(basis_state, basis_state′, args...)
        end
    end
    return val
end
export extend_operator

function operator_to_matrix(A, states)
    """
    Write an operator as a matrix in basis {states}.
    """
    n_states = length(states)
    A_mat = zeros(ComplexF64, n_states, n_states)
    for i in 1:n_states
        for j in 1:n_states
            A_mat[i,j] = extend_operator(A, states[i], states[j])
        end
    end
    return A_mat
end
export operator_to_matrix

function operator_to_matrix_zero_padding2(OA, A_states, B_states)
    """
    OA is an operator on Hilbert space A (basis = A_states).
    We would like to extend A to the direct-sum space A ⨁ B by padding with zeros, i.e.
    <i|OAB|j> = 0 if i∉A or j∉A, <i|OAB|j> = <i|OA|j> if i∈A and j∈A.
    """
    n_A = length(A_states)
    n_B = length(B_states)
    OAB_mat = zeros(ComplexF64, n_A+n_B, n_A+n_B)
    OAB_mat[1:n_A, 1:n_A] .= operator_to_matrix(OA, A_states)
    return OAB_mat
end
export operator_to_matrix_zero_padding2


