function update_H!(H, p, r, τ)
    @turbo for i in eachindex(H)
        H.re[i] = 0.0 #p.H₀.re[i]
        H.im[i] = 0.0 #p.H₀.im[i]
    end
    return nothing
end

function set_H_zero!(H)
    @turbo for i ∈ eachindex(H)
        H.re[i] = zero(eltype(H))
        H.im[i] = zero(eltype(H))
    end
    return nothing
end

function set_H_to_H₀!(H, H₀)
    @turbo for i ∈ eachindex(H)
        H.re[i] = H₀.re[i]
        H.im[i] = H₀.im[i]
    end
    return nothing
end

function update_H!(p, τ, r, fields, H, E_k, ds, ds_state1, ds_state2, Js)
 
    set_H_zero!(H)

    update_fields!(fields, r, τ)

    # Set summed fields to zero
    p.E -= p.E
    @inbounds @fastmath for i ∈ 1:3
        E_k[i] -= E_k[i]
    end
    
    # Sum updated fields
    @inbounds @simd for i ∈ eachindex(fields)
        E_i = fields.E[i] * sqrt(fields.s[i]) / (2 * √2)
        k_i = fields.k[i]
        p.E += E_i
        for k ∈ 1:3
            E_k[k] += E_i * k_i[k]
        end
    end

    # This can be used to exclude very off-resonant terms, need to add a user-defined setting to determine behavior
    # @inbounds for i ∈ eachindex(fields)
    #     s = fields.s[i]
    #     ω_field = fields.ω[i]
    #     x = sqrt(s) / (2 * √2)
    #     @inbounds for q ∈ 1:3
    #         E_i_q = fields.E[i][q]
    #         if norm(E_i_q) > 1e-10
    #             d_nnz_q = d_nnz[q]
    #             d_q = @view d[:,:,q]
    #             @inbounds @simd for cart_idx ∈ d_nnz_q
    #                 m, n = cart_idx.I
    #                 if abs(ω_field - (ω[n] - ω[m])) < 100 # Do not include very off-resonant terms (terms detuned by >10Γ)
    #                     val = x * E_i_q * d_q[m,n]
    #                     H[m,n] += val
    #                     H[n,m] += conj(val)
    #                 end
    #             end
    #         end
    #     end
    # end

    @inbounds @fastmath for q ∈ 1:3
        E_q = p.E[q]
        # E_q = conj(E_q) ####
        # if norm(E_q) > 1e-10
        E_q_re = real(E_q)
        E_q_im = imag(E_q)
        ds_q = ds[q]
        ds_q_re = ds_q.re
        ds_q_im = ds_q.im
        ds_state1_q = ds_state1[q]
        ds_state2_q = ds_state2[q]
        for i ∈ eachindex(ds_q)
            m = ds_state1_q[i] # excited state
            n = ds_state2_q[i] # ground state
            d_re = ds_q_re[i]
            d_im = ds_q_im[i]
            val_re = E_q_re * d_re - E_q_im * d_im
            val_im = E_q_re * d_im + E_q_im * d_re
            H.re[n,m] += -val_re # minus sign to make sure Hamiltonian is -d⋅E
            H.im[n,m] += -val_im
            H.re[m,n] += -val_re
            H.im[m,n] -= -val_im
        end
        # end
    end

    # Is this version faster?
    # @inbounds for q ∈ 1:3
    #     E_q = p.E[q]
    #     if norm(E_q) > 1e-10
    #         d_nnz_q = d_nnz[q]
    #         d_q = @view d[:,:,q]
    #         @inbounds @simd for cart_idx ∈ d_nnz_q
    #             m, n = cart_idx.I
    #             val = E_q * d_q[m,n]
    #             H[m,n] += val
    #             H[n,m] += conj(val)
    #         end
    #     end
    # end

    # diagonal terms like |e1><e1| for the non-hermitian part of H
    @inbounds @fastmath for J ∈ Js
        H.im[J.s, J.s] -= J.r # note that this is different from OBE calcs because we already converted to J.r = Γ/2
    end

    # off-diagonal terms like |e1><e2| for the non-hermitian part of H
    # they only exist when the ground state of the jump operator is the same (and have the same polarization)
    # @inbounds @fastmath for J ∈ Js
    #     for J′ ∈ Js
    #         if (J.s′ == J′.s′) && (J.q == J′.q)
    #             H.im[J.s, J′.s] -= sqrt(J.r) * sqrt(J′.r)
    #         end
    #     end
    # end

    return nothing
end
export update_H!

function update_H_nocomplex!(p, τ, r, fields, H_re, H_im, E_k_re, E_k_im, ds, ds_state1, ds_state2, Js)

    p.update_H(H_re, H_im, p, r, τ)
 
    update_fields!(fields, r, τ)

    # Set summed fields to zero
    p.E_re -= p.E_re
    p.E_im -= p.E_im
    @inbounds @fastmath for i ∈ 1:3
        E_k_re[i] -= E_k_re[i]
        E_k_im[i] -= E_k_im[i]
    end
    
    # Sum updated fields
    @inbounds @fastmath for i ∈ eachindex(fields)
        G = sqrt(fields.s[i]) / (2 * √2)
        E_i_re = G * real(fields.E[i])
        E_i_im = G * imag(fields.E[i])
        k_i = fields.k[i]
        p.E_re += E_i_re
        p.E_im += E_i_im
        for k ∈ 1:3
            E_k_re[k] += E_i_re * k_i[k]
            E_k_im[k] += E_i_im * k_i[k]
        end
    end

    @inbounds @fastmath for q ∈ 1:3
        E_q_re = p.E_re[q]
        E_q_im = p.E_im[q]
        ds_q = ds[q]
        ds_q_re = ds_q.re
        ds_q_im = ds_q.im
        ds_state1_q = ds_state1[q]
        ds_state2_q = ds_state2[q]
        for i ∈ eachindex(ds_q)
            m = ds_state1_q[i] # excited state
            n = ds_state2_q[i] # ground state
            d_re = ds_q_re[i]
            d_im = ds_q_im[i]
            val_re = E_q_re * d_re - E_q_im * d_im
            val_im = E_q_re * d_im + E_q_im * d_re
            H_re[n,m] += -val_re # minus sign to make sure Hamiltonian is -d⋅E
            H_im[n,m] += -val_im
            H_re[m,n] += -val_re
            H_im[m,n] -= -val_im
        end
    end

    @inbounds @fastmath for J ∈ Js
        H_im[J.s, J.s] -= J.r
    end

    return nothing
end
export update_H_nocomplex!

function update_H_obes!(p, τ, r, H₀, fields, H, E_k, ds, ds_state1, ds_state2, Js)

    # set_H_zero!(H)
    set_H_to_H₀!(H, H₀)
    
    # p.update_H(H, p, r, τ)

    update_fields!(fields, r, τ)

    # Set summed fields to zero
    p.E -= p.E
    @inbounds @fastmath for i ∈ 1:3
        E_k[i] -= E_k[i]
    end
    
    # Sum updated fields
    @inbounds @fastmath for i ∈ eachindex(fields)
        E_i = fields.E[i] * sqrt(fields.s[i]) / (2 * √2)
        k_i = fields.k[i]
        p.E += E_i
        for k ∈ 1:3
            E_k[k] += E_i * k_i[k]
        end
    end

    @inbounds @fastmath for q ∈ 1:3
        E_q = p.E[q]
        # E_q = conj(E_q) ####
        # if norm(E_q) > 1e-10
        E_q_re = real(E_q)
        E_q_im = imag(E_q)
        ds_q = ds[q]
        ds_q_re = ds_q.re
        ds_q_im = ds_q.im
        ds_state1_q = ds_state1[q]
        ds_state2_q = ds_state2[q]
        for i ∈ eachindex(ds_q)
            m = ds_state1_q[i] # excited state
            n = ds_state2_q[i] # ground state
            d_re = ds_q_re[i]
            d_im = ds_q_im[i]
            d_im = -d_im # added 8/23/24, making sure that Hamiltonian is -d*⋅E
            val_re = E_q_re * d_re - E_q_im * d_im
            val_im = E_q_re * d_im + E_q_im * d_re
            H.re[n,m] += -val_re # minus sign to make sure Hamiltonian is -d*⋅E
            H.im[n,m] += -val_im
            H.re[m,n] += -val_re
            H.im[m,n] -= -val_im
        end
        # end
    end

    # Is this version faster?
    # @inbounds for q ∈ 1:3
    #     E_q = p.E[q]
    #     if norm(E_q) > 1e-10
    #         d_nnz_q = d_nnz[q]
    #         d_q = @view d[:,:,q]
    #         @inbounds @simd for cart_idx ∈ d_nnz_q
    #             m, n = cart_idx.I
    #             val = E_q * d_q[m,n]
    #             H[m,n] += val
    #             H[n,m] += conj(val)
    #         end
    #     end
    # end

    @inbounds @fastmath for J ∈ Js
        H.im[J.s, J.s] -= norm(J.r)^2 / 2
    end

    # @inbounds @fastmath for J ∈ Js
    #     for J′ ∈ Js
    #         if (J.s′ == J′.s′) && (J.q == J′.q)
    #             H.im[J.s, J′.s] -= J.r * J′.r / 2
    #         end
    #     end
    # end

    # # remove very small terms (particularly, off-resonant terms)
    # for i ∈ eachindex(H)
    #     if norm(H[i]) < 1e-10
    #         H[i] = 0.0
    #     end
    # end

    return nothing
end
export update_H_obes!

    # This can be used to exclude very off-resonant terms, need to add a user-defined setting to determine behavior
    # @inbounds for i ∈ eachindex(fields)
    #     s = fields.s[i]
    #     ω_field = fields.ω[i]
    #     x = sqrt(s) / (2 * √2)
    #     @inbounds for q ∈ 1:3
    #         E_i_q = fields.E[i][q]
    #         if norm(E_i_q) > 1e-10
    #             d_nnz_q = d_nnz[q]
    #             d_q = @view d[:,:,q]
    #             @inbounds @simd for cart_idx ∈ d_nnz_q
    #                 m, n = cart_idx.I
    #                 if abs(ω_field - (ω[n] - ω[m])) < 100 # Do not include very off-resonant terms (terms detuned by >10Γ)
    #                     val = x * E_i_q * d_q[m,n]
    #                     H[m,n] += val
    #                     H[n,m] += conj(val)
    #                 end
    #             end
    #         end
    #     end
    # end