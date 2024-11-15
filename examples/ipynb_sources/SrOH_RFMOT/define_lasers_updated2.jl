function define_lasers(
        RF_frequency,
        lasers_tuple
    )

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
