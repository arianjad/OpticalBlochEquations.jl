function define_lasers(
        RF_frequency,
        lasers_tuple
    )

    lasers = []
    for laser ∈ lasers_tuple
        ω = laser[1]
        ϵ = laser[2]
        s = laser[3]
        k̂ = -(x̂ + ŷ) / sqrt(2)
        axes = [1,3]
        s_func = s_gaussian(s, axes; diagonal_xy=true)
        ϵ_func = ϵ_(rotate_pol(ϵ, k̂), rotate_pol(flip(ϵ), k̂))
        laser = Field(k̂, ϵ_func, ω, s_func)
        push!(lasers, laser)
    end

    return lasers
end
