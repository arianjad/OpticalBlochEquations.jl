using Pkg
Pkg.activate(".")
using Revise, QuantumStates, OpticalBlochEquations, UnitsToValue, DifferentialEquations, Plots, Statistics, LinearAlgebra, StaticArrays, DataFrames, CSV, ProgressMeter
using StaticArrays, RectiGrids, StatsBase, LoopVectorization, StructArrays
RESULTS_PATH = "./RESULTS_YbF"

###
function order_basis_by_m(basis)

    new_ordering = zeros(Int64, length(basis))

    # find the maximum value |M|
    m_max = 0
    for i ∈ eachindex(basis)
        if abs(basis[i].M) > m_max
            m_max = abs(basis[i].M)
        end
    end

    i = 1
    for m ∈ -m_max:m_max
        for j ∈ eachindex(basis)
            if basis[j].M == m
                new_ordering[i] = j
                i += 1
            end
        end
    end

    return basis[new_ordering]
end

function Zeeman_L(state::HundsCaseA_LinearMolecule, state′::HundsCaseA_LinearMolecule, p::Int64)
    """
    Electron orbital Zeeman interaction
    See Brown & Carrington, eq. (9.57)
    """
    v_1, v_2, ℓ, v_3, Λ, K, I, S, Σ, J, P, F, M = unpack(state)
    v_1′, v_2′, ℓ′, v_3′, Λ′, K′, I′, S′, Σ′, J′, P′, F′, M′ = unpack(state′)

    if ~delta(state, state′, :ℓ, :Λ, :K, :I, :S, :Σ, :P)
        return 0.0
    else
        return (
            (-1)^p * Λ * (-1)^(F - M + F′ + J + I + 1 + J - P) * wigner6j(J, F, I, F′, J′, 1) *
            wigner3j(F, 1, F′, -M, p, M′) * sqrt((2F + 1) * (2F′ + 1) * (2J + 1) * (2J′ + 1)) *
            wigner3j(J, 1, J′, -P, 0, P′)
        )
    end
end

function Zeeman_S(state::HundsCaseA_LinearMolecule, state′::HundsCaseA_LinearMolecule, p::Int64)
    """
    Electron spin Zeeman interaction
    See Brown & Carrington, eq. (9.58)
    """
    v_1, v_2, ℓ, v_3, Λ, K, I, S, Σ, J, P, F, M = unpack(state)
    v_1′, v_2′, ℓ′, v_3′, Λ′, K′, I′, S′, Σ′, J′, P′, F′, M′ = unpack(state′)

    if ~delta(state, state′, :ℓ, :I, :S)
        return 0.0
    else
        return (
            sum(
            (-1)^p * (-1)^(F - M + J + I + F′ + 1 + J - P + S - Σ) * wigner6j(J, F, I, F′, J′, 1) *
            wigner3j(F, 1, F′, -M, p, M′) * sqrt((2F + 1) * (2F′ + 1)) *
            wigner3j(J, 1, J′, -P, q, P′) * sqrt((2J + 1) * (2J′ + 1)) *
            wigner3j(S, 1, S, -Σ, q, Σ′) * sqrt(S * (S + 1) * (2S + 1))
            for q ∈ -1:1
        )
        )
    end
end

function Zeeman_gl′(state::HundsCaseA_LinearMolecule, state′::HundsCaseA_LinearMolecule, p::Int64)
    """

    """
    v_1, v_2, ℓ, v_3, Λ, K, I, S, Σ, J, P, F, M = unpack(state)
    v_1′, v_2′, ℓ′, v_3′, Λ′, K′, I′, S′, Σ′, J′, P′, F′, M′ = unpack(state′)

    if delta(state, state′)
        return 0.0
    else
        return (
            (-1)^p * (-1)^(F - M) * wigner3j_(F, 1, F′, -M, p, M′) * (-1)^(F′ + J + I + 1) * sqrt((2F + 1) * (2F′ + 1)) *
            wigner6j_(J′, F′, I, F, J, 1) *
            sum(
                δ(K′, K - 2q) * (-1)^(J - P + S - Σ) *
                (-1)^(J - P) * wigner3j_(J, 1, J′, -P, q, P′) * sqrt((2J + 1) * (2J′ + 1)) *
                (-1)^(S - Σ) * wigner3j_(S, 1, S, -Σ, -q, Σ′) * sqrt(S * (S + 1) * (2S + 1))
                for q ∈ (-1, 1)
            )
        )
    end
end




###
function run_scan(B, det, intensity, use_perp)
    # Quantum Number Bounds
    # Define the quantum number bounds for the system. Here, S, I, Λ, and N represent
    # the spin, nuclear spin, projection of the electronic orbital angular momentum
    # along the molecular axis, and the rotational quantum number, respectively.
    QN_bounds = (
        label="X",
        S=1 / 2,           # Spin quantum number
        I=1 / 2,           # Nuclear spin quantum number
        Λ=0,             # Projection of the electronic orbital angular momentum
        N=0:3            # Rotational quantum number range
    )

    # Generate all possible quantum states for a Hund's case (b) linear molecule
    X_state_basis = order_basis_by_m(enumerate_states(HundsCaseB_LinearMolecule, QN_bounds))

    # Define the Hamiltonian Operator
    X_state_operator = :(
        BX * Rotation +                     # Rotational energy term
        DX * RotationDistortion +           # Rotational distortion energy term
        γX * SpinRotation +                 # Spin-rotation interaction term
        bFX * Hyperfine_IS +                # Hyperfine interaction (Fermi contact term)
        cX * (Hyperfine_Dipolar / 3)        # Hyperfine interaction (dipolar term)
    )

    # Define the Parameters for the Hamiltonian
    # These parameters are specific to the YbF molecule and are given in Hertz (Hz).
    X_state_parameters = QuantumStates.@params begin
        BX = 7233.8271e6         # Rotational constant in Hz
        DX = 0.0                  # Rotational distortion constant in Hz (set to 0 for simplicity)
        γX = -13.41679e6         # Spin-rotation interaction constant in Hz
        bFX = 170.26374e6  # Fermi contact term in Hz
        cX = 85.4028e6     # Dipolar hyperfine interaction constant in Hz
    end

    # Create the Hamiltonian
    # Combine the basis states, operator, and parameters to define the Hamiltonian.
    X_state_ham = Hamiltonian(basis=X_state_basis, operator=X_state_operator, parameters=X_state_parameters)

    # Evaluate and Solve the Hamiltonian
    # Compute the energy levels and eigenstates of the Hamiltonian.
    evaluate!(X_state_ham)
    QuantumStates.solve!(X_state_ham)
    ground_states = X_state_ham.states[5:16] #subspace(X_state_ham.states,(N=1,))[1]


    # %% [markdown]
    # ### Create Hamiltonian for the $A^2\Pi_{1/2}(000, J=1/2+)$ state

    QN_bounds = (
        label="A",
        S=1 / 2,
        I=1 / 2,
        Λ=(-1, 1),
        J=1/2:3/2 # ?
    )
    A_state_basis = order_basis_by_m(enumerate_states(HundsCaseA_LinearMolecule, QN_bounds))

    A_state_operator = :(
        T_A * DiagonalOperator +
        Be_A * Rotation +
        Aso_A * SpinOrbit +
        q_A * ΛDoubling_q +
        p_A * ΛDoubling_p2q +
        a_A * Hyperfine_IL +
        d_A * Hyperfine_Dipolar_d
    )

    # Spectroscopic constants for YbF, A state
    A_state_parameters = QuantumStates.@params begin
        T_A = 563.27535e12   # Diagonal constant (electron zero point energy)
        Be_A = 0.2476292 * c * 1e2  # Rotational constant, taken from X state
        Aso_A = 1365.2908 * c * 1e2    # A spin-orbit constant
        p_A = -0.39707 * c * 1e2
        q_A = 0
        a_A = -1.6e6 / 2
        d_A = -4.6e6 #parity-dependent dipolar term
    end

    A_state_ham = Hamiltonian(basis=A_state_basis, operator=A_state_operator, parameters=A_state_parameters)
    evaluate!(A_state_ham)
    QuantumStates.solve!(A_state_ham)

    # Add Zeeman terms
    # Spectroscopic constants from "Optical Zeeman Spec... YbF" (Steimle and others)
    gL′ = 0.996 #1.18 #0.996 Two values possible from literature, their fits were inconclusive
    gl′ = -0.8016 #-0.722 #-0.8016

    M_x(M) = (state, state′) -> (M(state, state′, -1) - M(state, state′, +1)) / √2
    M_y(M) = (state, state′) -> im * (M(state, state′, -1) + M(state, state′, +1)) / √2
    M_z(M) = (state, state′) -> M(state, state′, 0)

    A_state_ham = add_to_H(A_state_ham, :B_x, (1e-4 * gS * μB / h) * M_x(Zeeman_S) + (1e-4 * gL′ * μB / h) * M_x(Zeeman_L) + (1e-4 * gl′ * μB / h) * M_x(Zeeman_gl′))
    A_state_ham = add_to_H(A_state_ham, :B_y, (1e-4 * gS * μB / h) * M_y(Zeeman_S) + (1e-4 * gL′ * μB / h) * M_y(Zeeman_L) + (1e-4 * gl′ * μB / h) * M_y(Zeeman_gl′))
    A_state_ham = add_to_H(A_state_ham, :B_z, (1e-4 * gS * μB / h) * M_z(Zeeman_S) + (1e-4 * gL′ * μB / h) * M_z(Zeeman_L) + (1e-4 * gl′ * μB / h) * M_z(Zeeman_gl′))

    A_state_ham.parameters.B_z = 0.0 #has to be flaot
    evaluate!(A_state_ham)
    QuantumStates.solve!(A_state_ham)

    # Now we can convert it to HundsCaseB
    excited_idx = subspace(A_state_ham.states, (J=1 / 2,))[1][5:8]
    excited_states = A_state_ham.states[excited_idx]
    QN_bounds = (
        S=1 / 2,
        I=1 / 2,
        Λ=(-1, 1),
        N=0:3
    )
    A_state_basis = order_basis_by_m(enumerate_states(HundsCaseB_LinearMolecule, QN_bounds))
    excited_states = convert_basis(excited_states, A_state_basis)

    states = [ground_states; excited_states]



    # %%
    # A few constants used for the simulation
    λ = 552e-9 # Wavelength of light in meters
    Γ = 2π * 5.7e6
    m = @with_unit 191 "u" # Mass of the molecule in atomic mass units
    k = 2π / λ

    ϵ(ϵ_val) = t -> ϵ_val
    s_func(s) = (r, t) -> s

    function define_lasers_XY_perp(states, ϕ)
        # note: ϕs are laser phases

        E1 = energy(states[1])
        E2 = energy(states[4])
        E3 = energy(states[10])

        E_excited = energy(states[16]) / 4 + 3 * energy(states[14]) / 4 #Center of mass

        detuning = +6Γ

        # Defining lasers as sidebands relative to lowest state, to match YbF paper
        ω1 = 2π * (E_excited - E1) + detuning
        ω2 = 2π * (E_excited - (E1 + 159e6)) + detuning
        ω3 = 2π * (E_excited - (E1 + 192e6)) + detuning

        ϵ(ϵ_val) = t -> ϵ_val
        s_func(s) = (r, t) -> s

        s = 100.0
        s1 = s_func(s / 2)
        s2 = s_func(s / 2)
        s3 = s_func(0.0 * s / 2)


        xpol = (-σ⁺ + σ⁻) / √2
        ypol = -im * (σ⁺ + σ⁻) / √2
        pol1x = σ⁰
        pol1y = xpol
        pol2x = σ⁰
        pol2y = xpol
        pol3x = σ⁰
        pol3y = xpol

        # note that phases are defined to be the same for each direction x and y
        k̂ = +x̂
        ϵ1 = ϵ(pol1x)
        laser1 = Field(k̂, ϵ1, ω1, s1)
        k̂ = -x̂
        ϵ2 = ϵ(pol1x)
        laser2 = Field(k̂, ϵ2, ω1, s1)
        k̂ = +ŷ
        ϵ3 = ϵ(exp(im * ϕ) * pol1y)
        laser3 = Field(k̂, ϵ3, ω1 + 2π * 1e6, s1)
        k̂ = -ŷ
        ϵ4 = ϵ(exp(im * ϕ) * pol1y)
        laser4 = Field(k̂, ϵ4, ω1 + 2π * 1e6, s1)

        lasers_XY_perp_1 = [laser1, laser2, laser3, laser4]

        k̂ = +x̂
        ϵ5 = ϵ(pol2x)
        laser5 = Field(k̂, ϵ5, ω2, s2)
        k̂ = -x̂
        ϵ6 = ϵ(pol2x)
        laser6 = Field(k̂, ϵ6, ω2, s2)
        k̂ = +ŷ
        ϵ7 = ϵ(exp(im * ϕ) * pol2y)
        laser7 = Field(k̂, ϵ7, ω2 + 2π * 1e6, s2)
        k̂ = -ŷ
        ϵ8 = ϵ(exp(im * ϕ) * pol2y)
        laser8 = Field(k̂, ϵ8, ω2 + 2π * 1e6, s2)

        lasers_XY_perp_2 = [laser5, laser6, laser7, laser8]

        k̂ = +x̂
        ϵ9 = ϵ(pol3x)
        laser9 = Field(k̂, ϵ9, ω3, s3)
        k̂ = -x̂
        ϵ10 = ϵ(pol3x)
        laser10 = Field(k̂, ϵ10, ω3, s3)
        k̂ = +ŷ
        ϵ11 = ϵ(exp(im * ϕ) * pol3y)
        laser11 = Field(k̂, ϵ11, ω3 + 2π * 1e6, s3)
        k̂ = -ŷ
        ϵ12 = ϵ(exp(im * ϕ) * pol3y)
        laser12 = Field(k̂, ϵ12, ω3 + 2π * 1e6, s3)

        lasers_XY_perp_3 = [laser9, laser10, laser11, laser12]

        lasers = [lasers_XY_perp_1; lasers_XY_perp_2; lasers_XY_perp_3]

        return lasers
    end
    function define_lasers_XY_par(states, ϕ)
        # note: ϕs are laser phases

        # alternative: E1 = energy(first(subspace(states, (Λ=0,F=1))[2]))
        E1 = energy(states[1])
        E2 = energy(states[4])
        E3 = energy(states[10])


        # Three ways of getting excited state position
        E_excited = energy(states[16]) / 4 + 3 * energy(states[14]) / 4 #Center of mass
        # E_excited = energy(states[16]) #Highest, F=0
        # E_excited = energy(states[13]) #Lowest, F=1


        detuning = +det * Γ

        # Defining lasers as sidebands relative to lowest state, to match YbF paper
        ω1 = 2π * (E_excited - E1) + detuning
        # ω2 = 2π * (E_excited - E2) + detuning
        ω2 = 2π * (E_excited - (E1 + 159e6)) + detuning
        # ω3 = 2π * (E_excited - E3) + detuning
        ω3 = 2π * (E_excited - (E1 + 192e6)) + detuning

        s = intensity
        s1 = s_func(s / 2)
        s2 = s_func(s / 2)
        s3 = s_func(0.0 * s / 2)

        pol1 = σ⁰
        pol2 = σ⁰
        pol3 = σ⁰

        # note that phases are defined to be the same for each direction x and y
        # Additional 1 MHz offset added to y beams relative to x to perform phase averaging
        k̂ = +x̂
        ϵ1 = ϵ(pol1)
        laser1 = Field(k̂, ϵ1, ω1, s1)
        k̂ = -x̂
        ϵ2 = ϵ(pol1)
        laser2 = Field(k̂, ϵ2, ω1, s1)
        k̂ = +ŷ
        ϵ3 = ϵ(exp(im * ϕ) * pol1)
        laser3 = Field(k̂, ϵ3, ω1 + 2π * 1e6, s1)
        k̂ = -ŷ
        ϵ4 = ϵ(exp(im * ϕ) * pol1)
        laser4 = Field(k̂, ϵ4, ω1 + 2π * 1e6, s1)

        lasers_XY_parallel_1 = [laser1, laser2, laser3, laser4]

        k̂ = +x̂
        ϵ5 = ϵ(pol2)
        laser5 = Field(k̂, ϵ5, ω2, s2)
        k̂ = -x̂
        ϵ6 = ϵ(pol2)
        laser6 = Field(k̂, ϵ6, ω2, s2)
        k̂ = +ŷ
        ϵ7 = ϵ(exp(im * ϕ) * pol2)
        laser7 = Field(k̂, ϵ7, ω2 + 2π * 1e6, s2)
        k̂ = -ŷ
        ϵ8 = ϵ(exp(im * ϕ) * pol2)
        laser8 = Field(k̂, ϵ8, ω2 + 2π * 1e6, s2)

        lasers_XY_parallel_2 = [laser5, laser6, laser7, laser8]

        k̂ = +x̂
        ϵ9 = ϵ(pol3)
        laser9 = Field(k̂, ϵ9, ω3, s3)
        k̂ = -x̂
        ϵ10 = ϵ(pol3)
        laser10 = Field(k̂, ϵ10, ω3, s3)
        k̂ = +ŷ
        ϵ11 = ϵ(exp(im * ϕ) * pol3)
        laser11 = Field(k̂, ϵ11, ω3 + 2π * 1e6, s3)
        k̂ = -ŷ
        ϵ12 = ϵ(exp(im * ϕ) * pol3)
        laser12 = Field(k̂, ϵ12, ω3 + 2π * 1e6, s3)

        lasers_XY_parallel_3 = [laser9, laser10, laser11, laser12]

        lasers = [lasers_XY_parallel_1; lasers_XY_parallel_2; lasers_XY_parallel_3]

        return lasers
    end

    if use_perp
        lasers = define_lasers_XY_perp(states, 0.0)
    else
        lasers = define_lasers_XY_par(states, 0.0)
    end


    # %%
    evaluate!(X_state_ham)
    QuantumStates.solve!(X_state_ham)
    ground_states = X_state_ham.states[5:16]

    states = [ground_states; excited_states]

    # calculate transitions dipole moments
    d = zeros(ComplexF64, 16, 16, 3)
    d_ge = zeros(ComplexF64, 12, 4, 3)
    basis_tdms = get_tdms_two_bases(ground_states[1].basis, excited_states[1].basis, TDM)
    tdms_between_states!(d_ge, basis_tdms, ground_states, excited_states)
    d[1:12, 13:16, :] .= d_ge


    Zeeman_x(state, state′) = (Zeeman(state, state′, -1) - Zeeman(state, state′, 1)) / sqrt(2)
    Zeeman_y(state, state′) = im * (Zeeman(state, state′, -1) + Zeeman(state, state′, 1)) / sqrt(2)
    Zeeman_z(state, state′) = Zeeman(state, state′, 0)

    Zeeman_x_mat = StructArray(operator_to_matrix_zero_padding2(Zeeman_x, ground_states, excited_states) .* (1e-4 * gS * μB * (2π / Γ) / h)) # magnetic field in units of G
    Zeeman_y_mat = StructArray(operator_to_matrix_zero_padding2(Zeeman_y, ground_states, excited_states) .* (1e-4 * gS * μB * (2π / Γ) / h))
    Zeeman_z_mat = StructArray(operator_to_matrix_zero_padding2(Zeeman_z, ground_states, excited_states) .* (1e-4 * gS * μB * (2π / Γ) / h))

    # let's add effective Zeeman terms for the A state
    Zeeman_A_x = (1e-4 * gS * μB / h) * M_x(Zeeman_S) + (1e-4 * gL′ * μB / h) * M_x(Zeeman_L) + (1e-4 * gl′ * μB / h) * M_x(Zeeman_gl′)
    Zeeman_A_y = (1e-4 * gS * μB / h) * M_y(Zeeman_S) + (1e-4 * gL′ * μB / h) * M_y(Zeeman_L) + (1e-4 * gl′ * μB / h) * M_y(Zeeman_gl′)
    Zeeman_A_z = (1e-4 * gS * μB / h) * M_z(Zeeman_S) + (1e-4 * gL′ * μB / h) * M_z(Zeeman_L) + (1e-4 * gl′ * μB / h) * M_z(Zeeman_gl′)

    Zeeman_x_mat[13:16, 13:16] .= (2π / Γ) .* operator_to_matrix(Zeeman_A_x, A_state_ham.states[13:16])
    Zeeman_y_mat[13:16, 13:16] .= (2π / Γ) .* operator_to_matrix(Zeeman_A_y, A_state_ham.states[13:16])
    Zeeman_z_mat[13:16, 13:16] .= (2π / Γ) .* operator_to_matrix(Zeeman_A_z, A_state_ham.states[13:16])

    function update_H_and_∇H(H, p, r, t)

        Zeeman_Hz = p.extra_data.Zeeman_Hz
        Zeeman_Hx = p.extra_data.Zeeman_Hx
        Zeeman_Hy = p.extra_data.Zeeman_Hy

        Bx = p.sim_params.Bx
        By = p.sim_params.By
        Bz = p.sim_params.Bz

        @turbo for i in eachindex(H)
            H.re[i] = Bx * Zeeman_Hx.re[i] + By * Zeeman_Hy.re[i] + Bz * Zeeman_Hz.re[i]
            H.im[i] = Bx * Zeeman_Hx.im[i] + By * Zeeman_Hy.im[i] + Bz * Zeeman_Hz.im[i]
        end

        ∇H = SVector{3,ComplexF64}(0, 0, 0)

        return ∇H
    end

    sim_params = (Bx=0, By=B / √2, Bz=B / √2)
    extra_data = (
        Zeeman_Hx=Zeeman_x_mat,
        Zeeman_Hy=Zeeman_y_mat,
        Zeeman_Hz=Zeeman_z_mat,
        states_static=states
    )


    # %%
    # Set initial conditions
    particle = Particle()
    ρ0 = zeros(ComplexF64, length(states), length(states))
    # ρ0[13, 13] = 1.0
    for n_ground in 1:12
        ρ0[n_ground, n_ground] = 1.0 / 12
    end


    freq_res = 1e-2
    p = obe(ρ0, particle, states, lasers, d, true, true; λ=λ, Γ=Γ, freq_res=freq_res, extra_data=extra_data, sim_params=sim_params, update_H_and_∇H=update_H_and_∇H)


    # %%
    p.r0 = (0, 0.0, 0.0) ./ (1 / k)
    p.v = (0.0, 1.0, 0.0) ./ (Γ / k)
    p.v = round_vel(p.v, p.freq_res)


    # %%
    t_end = 10p.period
    tspan = (0.0, t_end)
    prob = ODEProblem(ρ!, p.ρ0_vec, tspan, p)
    times = range(0, t_end, 10000)

    cb = PeriodicCallback(reset_force!, p.period)
    @time sol = DifferentialEquations.solve(prob; alg=DP5(), callback=cb, reltol=1e-5, saveat=times)

    # Print the force
    print("Force (10³ m/s): ", sol.prob.p.force_last_period * (1e-3 * ħ * k * Γ / m))


    # %%
    prob.p.ρ_soa |> tr

    # %%
    function prob_func!(prob, scan_values_grid, i)

        p = prob.p

        particle = Particle()
        dir = scan_values_grid[i].dir
        particle.v .= (scan_values_grid[i].v * dir[1], scan_values_grid[i].v * dir[2], 0)
        # particle.v .= (0, scan_values_grid[i].v, 0)

        states = p.extra_data.states_static
        lasers = define_lasers_XY_par(states, scan_values_grid[i].ϕ)

        d = p.d
        extra_data = p.extra_data
        sim_params = p.sim_params

        # create new problem
        freq_res = 1e-2
        callback = PeriodicCallback(reset_force!, p.period)
        p = obe(ρ0, particle, states, lasers, d, true, true; λ=λ, Γ=Γ, freq_res=freq_res, extra_data=extra_data, sim_params=sim_params, update_H_and_∇H=update_H_and_∇H)
        prob′ = ODEProblem(ρ!, p.ρ0_vec, tspan, p; callback=callback, reltol=1e-5, save_on=false)

        return prob′
    end
    function output_func(p, sol)
        f = p.force_last_period
        return (f[1], f[2], f[3])
    end


    # %%
    freq_res = 1e-2

    particle.v .= (0, 1.0, 0)
    p = obe(ρ0, particle, states, lasers, d, true, true; λ=λ, Γ=Γ, freq_res=freq_res, extra_data=extra_data, sim_params=sim_params, update_H_and_∇H=update_H_and_∇H)

    t_end = 10p.period
    tspan = (0.0, t_end)

    prob = ODEProblem(ρ!, p.ρ0_vec, tspan, p, reltol=1e-5, save_on=false)

    scan_values = (
        v=(0:0.5:6) ./ (Γ / k),
        # ϕ=range(0, π, 8),
        ϕ=[0.0],
        dir=[(cos(θ), sin(θ), 0) for θ ∈ range(0, 2π, 8)]
    )
    scan_values_grid = RectiGrids.grid(scan_values)


    # %%
    @time forces, populations = force_scan_v2(prob, scan_values_grid, prob_func!, output_func)


    # %%
    averaged_forces = Float64[]
    stds_forces = Float64[]
    for (i, v) ∈ enumerate(scan_values.v)
        idxs = [j for (j, x) ∈ enumerate(scan_values_grid) if x.v == v]
        _forces = [forces[idx] ⋅ scan_values_grid[idx].dir for idx in idxs] .* (1e-3 * ħ * k * Γ / m)
        push!(averaged_forces, mean(_forces))
        push!(stds_forces, std(_forces))
    end

    # %%
    flag = use_perp ? "perp" : "par"
    path = "$(RESULTS_PATH)/$(flag)_B_$(B)_detuning_$(det)_intensity_$(intensity)"
    println("Saving to $path")
    df = DataFrame(
        v=scan_values.v,
        a=averaged_forces,
        std=stds_forces
    )
    CSV.write(path * ".csv", df)
    return df
end
