function flip(ϵ)
    return SVector{3, ComplexF64}(ϵ[3],ϵ[2],ϵ[1])
end

function gaussian_intensity_along_axes_updated(r, axes; diagonal_xy=false)
    """1/e^2 width = 10 mm Gaussian beam """
    if diagonal_xy
        d2 = (abs(r[1] - r[2]) / sqrt(2))^2 + r[3]^2
    else
        d2 = r[axes[1]]^2 + r[axes[2]]^2
    end
    return exp(-2*d2/(20e-3*k)^2) * (sqrt(d2) < 20e-3*k)
end

ϵ_(ϵ1, ϵ2) = t -> iseven(t ÷ (π/RF_frequency)) ? ϵ1 : ϵ2
s_func(s) = (x,t) -> s
s_gaussian(s, axes; diagonal_xy=false) = (r,t) -> s * gaussian_intensity_along_axes_updated(r, axes)

function define_lasers(
        RF_frequency,
        lasers_tuple
    )

    # lasers = []
    # for laser ∈ lasers_tuple
    #     ω = laser[1]
    #     ϵ = laser[2]
    #     s = laser[3]
    #     for (i,k̂) ∈ enumerate((x̂,ŷ,ẑ))
    #         axes = setdiff((1,2,3), i)
    #         if k̂ == ẑ
    #             ϵ = flip(ϵ)
    #         end            
    #         for k_dir ∈ (+1,-1)
    #             s_func = s_gaussian(s, axes)
    #             ϵ_func = ϵ_(rotate_pol(ϵ, k_dir .* k̂), rotate_pol(flip(ϵ), k_dir .* k̂))
    #             laser = Field(k_dir .* k̂, ϵ_func, ω, s_func)
    #             push!(lasers, laser)
    #         end
    #     end
    # end
    
    lasers = []
    for laser ∈ lasers_tuple
        ω = laser[1]
        ϵ = laser[2]
        s = laser[3]
        for (i,k̂) ∈ enumerate((x̂,ŷ))
            axes = setdiff((1,2,3), i)
            if k̂ == ẑ
                ϵ = flip(ϵ)
            end            
            for k_dir ∈ (-1)
                s_func = s_gaussian(s, axes)
                ϵ_func = ϵ_(rotate_pol(ϵ, k_dir .* k̂), rotate_pol(flip(ϵ), k_dir .* k̂))
                laser = Field(k_dir .* k̂, ϵ_func, ω, s_func)
                push!(lasers, laser)
            end
        end
    end

    return lasers
end
