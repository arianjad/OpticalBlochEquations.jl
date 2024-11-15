function flip(ϵ)
    return SVector{3, ComplexF64}(ϵ[3],ϵ[2],ϵ[1])
end

function gaussian_intensity_along_axes(r, axes, centers)
    """1/e^2 width = 10 mm Gaussian beam """
    d2 = (r[axes[1]] - centers[1])^2 + (r[axes[2]] - centers[2])^2   
    return exp(-2*d2/(20e-3/(1/k))^2)
end

# function gaussian_intensity_along_axes(r, axes, centers) # flat top beam
#     """ 1/e^2 width = 10 mm Gaussian beam """
#     d2 = (r[axes[1]] - centers[1])^2 + (r[axes[2]] - centers[2])^2   
#     return sqrt(d2) <= 2*10e-3/(1/k)
# end

function define_lasers(
        states,
        RF_frequency,
        s1,
        s2,
        s3,
        s4,
        Δ1,
        Δ2,
        Δ3,
        Δ4,
        pol1_x,
        pol2_x,
        pol3_x,
        pol4_x,
        pol_imbalance,
        s_imbalance,
        retro_loss,
        off_center,
        pointing_error,
        s5,
        s6,
        Δ5,
        Δ6,
        pol5_x,
        pol6_x
    )
    
    ω1 = 2π * (energy(states[end]) - energy(states[1])) + Δ1
    ω2 = 2π * (energy(states[end]) - energy(states[5])) + Δ2
    ω3 = 2π * (energy(states[end]) - energy(states[1])) + Δ3
    ω4 = 2π * (energy(states[end]) - energy(states[5])) + Δ4
    
    x_center_y = rand() * off_center[1] * k
    x_center_z = rand() * off_center[2] * k
    y_center_x = rand() * off_center[3] * k
    y_center_z = rand() * off_center[4] * k
    z_center_x = rand() * off_center[5] * k
    z_center_y = rand() * off_center[6] * k
    
    ϵ_(ϵ1, ϵ2) = t -> iseven(t ÷ (π/RF_frequency)) ? ϵ1 : ϵ2
    s_func(s) = (x,t) -> s
    s_gaussian(s, axes, centers) = (r,t) -> s * gaussian_intensity_along_axes(r, axes, centers)
    
    rand1 = rand()
    pol1_x = pol1_x.*sqrt(1 - rand1*pol_imbalance) + flip(pol1_x).*sqrt(rand1*pol_imbalance)
    rand2 = rand()
    pol2_x = pol2_x.*sqrt(1 - rand2*pol_imbalance) + flip(pol2_x).*sqrt(rand2*pol_imbalance)
    rand3 = rand()
    pol3_x = pol3_x.*sqrt(1 - rand3*pol_imbalance) + flip(pol3_x).*sqrt(rand3*pol_imbalance)
    rand4 = rand()
    pol4_x = pol4_x.*sqrt(1 - rand4*pol_imbalance) + flip(pol4_x).*sqrt(rand4*pol_imbalance)
    rand4 = rand()
    
    sx_rand = 1/2-rand()
    sy_rand = 1/2-rand()
    sz_rand = 1/2-rand()
    
    # ϕs = [exp(im*2π*rand()),exp(im*2π*rand()),exp(im*2π*rand()),exp(im*2π*rand()),exp(im*2π*rand()),exp(im*2π*rand())]
    
    ϕs = [1,1,1,1,1,1]
    
    kx = x̂ + [0, pointing_error[1],pointing_error[2]]
    kx = kx ./ sqrt(kx[1]^2 + kx[2]^2 + kx[3]^2)
    ky = ŷ + [pointing_error[3],0.0,pointing_error[4]]
    ky = ky ./ sqrt(ky[1]^2 + ky[2]^2 + ky[3]^2)
    kz = ẑ + [pointing_error[5],pointing_error[6],0.0]
    kz = kz / sqrt(kz[1]^2 + kz[2]^2 + kz[3]^2)
    
    s1x = s1 * (1+s_imbalance[1]*sx_rand)
    s1y = s1 * (1+s_imbalance[2]*sy_rand)
    s1z = s1 * (1+s_imbalance[3]*sz_rand)
    
    k̂1 = +kx; 
    k̂2 = -kx; 
    k̂3 = +ky; 
    k̂4 = -ky; 
    k̂5 = +kz; 
    k̂6 = -kz; 
    
    ϵ1 = ϕs[1]*rotate_pol(pol1_x, k̂1)
    ϵ2 = ϕs[2]*rotate_pol(pol1_x, k̂2)
    ϵ3 = ϕs[3]*rotate_pol(pol1_x, k̂3)
    ϵ4 = ϕs[4]*rotate_pol(pol1_x, k̂4)
    ϵ5 = ϕs[5]*rotate_pol(flip(pol1_x), k̂5)
    ϵ6 = ϕs[6]*rotate_pol(flip(pol1_x), k̂6)
    
    k̂1 = +kx; ϵ_func1 = ϵ_(ϵ1, ϵ2); laser1 = Field(k̂1, ϵ_func1, ω1, s_gaussian(s1x, (2,3), (x_center_y, x_center_z)))
    k̂2 = -kx; ϵ_func2 = ϵ_(ϵ2, ϵ1); laser2 = Field(k̂2, ϵ_func2, ω1, s_gaussian(s1x*(1-retro_loss), (2,3), (x_center_y, x_center_z)))
    k̂3 = +ky; ϵ_func3 = ϵ_(ϵ3, ϵ4); laser3 = Field(k̂3, ϵ_func3, ω1, s_gaussian(s1y, (1,3), (y_center_x, y_center_z)))
    k̂4 = -ky; ϵ_func4 = ϵ_(ϵ4, ϵ3); laser4 = Field(k̂4, ϵ_func4, ω1, s_gaussian(s1y*(1-retro_loss), (1,3), (y_center_x, y_center_z)))
    k̂5 = +kz; ϵ_func5 = ϵ_(ϵ5, ϵ6); laser5 = Field(k̂5, ϵ_func5, ω1, s_gaussian(s1z, (1,2), (z_center_x, z_center_y)))
    k̂6 = -kz; ϵ_func6 = ϵ_(ϵ6, ϵ5); laser6 = Field(k̂6, ϵ_func6, ω1, s_gaussian(s1z*(1-retro_loss), (1,2), (z_center_x, z_center_y)))

    lasers_1 = [laser1, laser2, laser3, laser4, laser5, laser6]
    
    s2x = s2 * (1+s_imbalance[1]*sx_rand)
    s2y = s2 * (1+s_imbalance[2]*sy_rand)
    s2z = s2 * (1+s_imbalance[3]*sz_rand)
    
    k̂7 = +kx; 
    k̂8 = -kx; 
    k̂9 = +ky; 
    k̂10 = -ky; 
    k̂11 = +kz; 
    k̂12 = -kz; 
    
    ϵ7 = ϕs[1]*rotate_pol(pol2_x, k̂7)
    ϵ8 = ϕs[2]*rotate_pol(pol2_x, k̂8)
    ϵ9 = ϕs[3]*rotate_pol(pol2_x, k̂9)
    ϵ10 = ϕs[4]*rotate_pol(pol2_x, k̂10)
    ϵ11 = ϕs[5]*rotate_pol(flip(pol2_x), k̂11)
    ϵ12 = ϕs[6]*rotate_pol(flip(pol2_x), k̂12)
    
    ϵ_func7 = ϵ_(ϵ7, ϵ8); laser7 = Field(k̂7, ϵ_func7, ω2, s_gaussian(s2x, (2,3), (x_center_y, x_center_z)))
    ϵ_func8 = ϵ_(ϵ8, ϵ7); laser8 = Field(k̂8, ϵ_func8, ω2, s_gaussian(s2x*(1-retro_loss), (2,3), (x_center_y, x_center_z)))
    ϵ_func9 = ϵ_(ϵ9, ϵ10); laser9 = Field(k̂9, ϵ_func9, ω2, s_gaussian(s2y, (1,3), (y_center_x, y_center_z)))
    ϵ_func10 = ϵ_(ϵ10, ϵ9); laser10 = Field(k̂10, ϵ_func10, ω2, s_gaussian(s2y*(1-retro_loss),  (1,3), (y_center_x, y_center_z)))
    ϵ_func11 = ϵ_(ϵ11, ϵ12); laser11 = Field(k̂11, ϵ_func11, ω2, s_gaussian(s2z, (1,2), (z_center_x, z_center_y)))
    ϵ_func12 = ϵ_(ϵ12, ϵ11); laser12 = Field(k̂12, ϵ_func12, ω2, s_gaussian(s2z*(1-retro_loss), (1,2), (z_center_x, z_center_y)))

    lasers_2 = [laser7, laser8, laser9, laser10, laser11, laser12]

    s3x = s3 * (1+s_imbalance[1]*sx_rand)
    s3y = s3 * (1+s_imbalance[2]*sy_rand)
    s3z = s3 * (1+s_imbalance[3]*sz_rand)
    
    k̂13 = +kx; 
    k̂14 = -kx; 
    k̂15 = +ky; 
    k̂16 = -ky; 
    k̂17 = +kz; 
    k̂18 = -kz; 
    
    ϵ13 = ϕs[1]*rotate_pol(pol3_x, k̂13)
    ϵ14 = ϕs[2]*rotate_pol(pol3_x, k̂14)
    ϵ15 = ϕs[3]*rotate_pol(pol3_x, k̂15)
    ϵ16 = ϕs[4]*rotate_pol(pol3_x, k̂16)
    ϵ17 = ϕs[5]*rotate_pol(flip(pol3_x), k̂17)
    ϵ18 = ϕs[6]*rotate_pol(flip(pol3_x), k̂18)
    
    ϵ_func13 = ϵ_(ϵ13, ϵ14); laser13 = Field(k̂13, ϵ_func13, ω3, s_gaussian(s3x, (2,3), (x_center_y, x_center_z)))
    ϵ_func14 = ϵ_(ϵ14, ϵ13); laser14 = Field(k̂14, ϵ_func14, ω3, s_gaussian(s3x*(1-retro_loss), (2,3), (x_center_y, x_center_z)))
    ϵ_func15 = ϵ_(ϵ15, ϵ16); laser15 = Field(k̂15, ϵ_func15, ω3, s_gaussian(s3y, (1,3), (y_center_x, y_center_z)))
    ϵ_func16 = ϵ_(ϵ16, ϵ15); laser16 = Field(k̂16, ϵ_func16, ω3, s_gaussian(s3y*(1-retro_loss), (1,3), (y_center_x, y_center_z)))
    ϵ_func17 = ϵ_(ϵ17, ϵ18); laser17 = Field(k̂17, ϵ_func17, ω3, s_gaussian(s3z, (1,2), (z_center_x, z_center_y)))
    ϵ_func18 = ϵ_(ϵ18, ϵ17); laser18 = Field(k̂18, ϵ_func18, ω3, s_gaussian(s3z*(1-retro_loss), (1,2), (z_center_x, z_center_y)))

    lasers_3 = [laser13, laser14, laser15, laser16, laser17, laser18]

    k̂19 = +kx 
    k̂20 = -kx 
    k̂21 = +ky 
    k̂22 = -ky 
    k̂23 = +kz
    k̂24 = -kz
    
    s4x = s4 * (1+s_imbalance[1]*sx_rand)
    s4y = s4 * (1+s_imbalance[2]*sy_rand)
    s4z = s4 * (1+s_imbalance[3]*sz_rand)
    
    ϵ19 = ϕs[1]*rotate_pol(pol4_x, k̂19)
    ϵ20 = ϕs[2]*rotate_pol(pol4_x, k̂20)
    ϵ21 = ϕs[3]*rotate_pol(pol4_x, k̂21)
    ϵ22 = ϕs[4]*rotate_pol(pol4_x, k̂22)
    ϵ23 = ϕs[5]*rotate_pol(flip(pol4_x), k̂23)
    ϵ24 = ϕs[6]*rotate_pol(flip(pol4_x), k̂24)
    
    ϵ_func19 = ϵ_(ϵ19, ϵ20); laser19 = Field(k̂19, ϵ_func19, ω4, s_gaussian(s4x, (2,3), (x_center_y, x_center_z)))
    ϵ_func20 = ϵ_(ϵ20, ϵ19); laser20 = Field(k̂20, ϵ_func20, ω4, s_gaussian(s4x*(1-retro_loss), (2,3), (x_center_y, x_center_z)))
    ϵ_func21 = ϵ_(ϵ21, ϵ22); laser21 = Field(k̂21, ϵ_func21, ω4, s_gaussian(s4y, (1,3), (y_center_x, y_center_z)))
    ϵ_func22 = ϵ_(ϵ22, ϵ21); laser22 = Field(k̂22, ϵ_func22, ω4, s_gaussian(s4y*(1-retro_loss), (1,3), (y_center_x, y_center_z)))
    ϵ_func23 = ϵ_(ϵ23, ϵ24); laser23 = Field(k̂23, ϵ_func23, ω4, s_gaussian(s4z, (1,2), (z_center_x, z_center_y)))
    ϵ_func24 = ϵ_(ϵ24, ϵ23); laser24 = Field(k̂24, ϵ_func24, ω4, s_gaussian(s4z*(1-retro_loss), (1,2), (z_center_x, z_center_y)))
    
    lasers_4 = [laser19, laser20, laser21, laser22, laser23, laser24]

    # create "slowing" lasers that sit "behind" the MOT and pushes molecules back into the MOT
    ω5 = 2π * (energy(states[end]) - energy(states[1])) + Δ5
    ω6 = 2π * (energy(states[end]) - energy(states[5])) + Δ6
    
    slowing_beams_displacement = 12e-3 / (1 / k)

    k̂25 = -kx
    k̂26 = -ky
    
    ϵ25 = ϕs[1]*rotate_pol(pol5_x, k̂25)
    ϵ26 = ϕs[2]*rotate_pol(pol5_x, k̂26)
    
    ϵ_func25 = ϵ_(ϵ25, ϵ26); laser25 = Field(k̂25, ϵ_func25, ω5, s_gaussian(s5, (2,3), (slowing_beams_displacement, 0.)))
    ϵ_func26 = ϵ_(ϵ26, ϵ25); laser26 = Field(k̂26, ϵ_func26, ω5, s_gaussian(s5, (1,3), (slowing_beams_displacement, 0.)))

    k̂29 = -kx
    k̂30 = -ky
    
    ϵ29 = ϕs[1]*rotate_pol(pol6_x, k̂29)
    ϵ30 = ϕs[2]*rotate_pol(pol6_x, k̂30)
    
    ϵ_func29 = ϵ_(ϵ29, ϵ30); laser29 = Field(k̂29, ϵ_func29, ω6, s_gaussian(s6, (2,3), (slowing_beams_displacement, 0.)))
    ϵ_func30 = ϵ_(ϵ30, ϵ29); laser30 = Field(k̂30, ϵ_func30, ω6, s_gaussian(s6, (1,3), (slowing_beams_displacement, 0.)))
    
    slowing_lasers = [laser25, laser26, laser29, laser30]
    
    lasers = [lasers_1; lasers_2; lasers_3; lasers_4; slowing_lasers]

end
