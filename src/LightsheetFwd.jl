module LightSheetSimulation

export simulate_lightsheet_psf, simulate_lightsheet_image

using PointSpreadFunctions, LinearAlgebra, FourierTools, NDTools

function simulate_lightsheet_psf(sz, pp_illu, pp_det, sampling, max_components)

    # illumination psf:
    h2d = sum(abs2.(sum(apsf(sz, pp_illu; sampling=sampling), dims=1)), dims=4)[1, :, :, 1]
    otf2d = fft(h2d)

    # detection psf:
    h_det = psf(sz, pp_det; sampling=sampling)

    F = svd(otf2d)

    otf_comp_s = Vector{Any}(undef, max_components)
    otf_comp_t = Vector{Any}(undef, max_components)
    otf_comp_s[1]= (F.S[1] * F.Vt[1, :])'
    otf_comp_t[1]= F.U[:, 1]
    for n in 2:max_components
        otf_comp_s[n]= (F.S[n] * F.Vt[n, :])'
        otf_comp_t[n]= F.U[:, n]
    end

    
    psf_comp_t = [real.(ifft(Array(otf_comp_t[n]), 1)) for n in 1:max_components]
    psf_comp_s = [real.(ifft(Array(otf_comp_s[n]), 2)) for n in 1:max_components]

    return psf_comp_t, psf_comp_s, h_det
end

function simulate_lightsheet_image(obj::AbstractArray{T, N}, sz, psf_comp_t, psf_comp_s, h_det, fwd_components) where {T, N}

    lightsheet_img = zeros(eltype(obj), sz)

    for n in 1:fwd_components
         psf_det_comb = reorient(psf_comp_t[n], Val(3)) .* h_det
         lightsheet_img += conv_psf(obj .* reorient(psf_comp_s[n], Val(1)), psf_det_comb)
         end
    return lightsheet_img
end

end # module