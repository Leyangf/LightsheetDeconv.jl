module LightSheetSimulation
    export simulate_lightsheet_psf, simulate_lightsheet_image
    using PointSpreadFunctions, LinearAlgebra, FourierTools, NDTools

    function simulate_lightsheet_psf(sz, pp_ill, sampling, max_components=4)

        h_ill = sum(abs2.(sum(apsf(sz, pp_ill; sampling=sampling), dims=1)), dims=4)[1, :, :, 1]
    
        F = svd(h_ill)
    
        psf_comp_x = Vector{Any}(undef, max_components)
        psf_comp_z = Vector{Any}(undef, max_components)
        for n in 1:max_components
            psf_comp_x[n]= (F.S[n] * F.Vt[n, :])'
            psf_comp_z[n]= F.U[:, n]
        end
    
        psf_comp_x = [psf_comp_x[n] for n in 1:max_components]
        psf_comp_z = [psf_comp_z[n] for n in 1:max_components]
    
        return psf_comp_x, psf_comp_z
    end

    function simulate_lightsheet_image(obj::AbstractArray{T, N}, sz, psf_comp_x, psf_comp_z, h_det, fwd_components=2) where {T, N}

        lightsheet_img = zeros(eltype(obj), sz)
    
        for n in 1:fwd_components
             psf_det_comb = reorient(psf_comp_z[n], Val(3)) .* h_det
             lightsheet_img += conv_psf(obj .* reorient(psf_comp_x[n], Val(1)), psf_det_comb)
             end
        return lightsheet_img
    end

end # module
