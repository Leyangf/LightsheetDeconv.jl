module LightSheetSimulation
    export simulate_lightsheet_psf, simulate_lightsheet_image
    using PointSpreadFunctions, LinearAlgebra, FourierTools, NDTools

    # ----------------------------------------------------------------------
    # Function: simulate_lightsheet_psf
    # Description: Simulates the Point Spread Function (PSF) for light sheet microscopy.
    # Arguments:
    #   - sz: Size of the 3D volume.
    #   - pp_illu: Illumination PSF parameters.
    #   - pp_det: Detection PSF parameters.
    #   - sampling: Sampling rate (spatial resolution).
    #   - max_components: Number of components for SVD decomposition.
    #   - dslm: Boolean flag for using digitally scanned light sheet microscopy (default: false).
    # ----------------------------------------------------------------------
    function simulate_lightsheet_psf(sz, pp_illu, pp_det, sampling, max_components, dslm=false)
        # Compute illumination PSF
        if !(dslm)
            # For non-DSLM, calculate coherent PSF and sum along x-dimension
            h2d = sum(abs2.(sum(apsf(sz, pp_illu; sampling=sampling), dims=1)), dims=4)[1, :, :, 1]
        else
            # For DSLM, sum incoherent PSF along x-dimension
            h2d = sum(hpsf(sz, pp_illu; sampling=(0.20, 0.2, 0.2)), dims=1)[1, :, :]
        end

        # Compute 2D OTF (Optical Transfer Function) via FFT
        otf2d = fft(h2d)

        # Compute detection PSF
        h_det = psf(sz, pp_det; sampling=sampling)

        # Perform Singular Value Decomposition (SVD) on OTF
        F = svd(otf2d)

        # Initialize storage for spatial and temporal PSF components
        otf_comp_s = Vector{Any}(undef, max_components)
        otf_comp_t = Vector{Any}(undef, max_components)

        # Compute the first component
        otf_comp_s[1] = (F.S[1] * F.Vt[1, :])'
        otf_comp_t[1] = F.U[:, 1]

        # Compute subsequent components based on SVD
        for n in 2:max_components
            otf_comp_s[n] = (F.S[n] * F.Vt[n, :])'
            otf_comp_t[n] = F.U[:, n]
        end

        # Convert OTF components back to PSF components via inverse FFT
        psf_comp_t = [real.(ifft(Array(otf_comp_t[n]), 1)) for n in 1:max_components]
        psf_comp_s = [real.(ifft(Array(otf_comp_s[n]), 2)) for n in 1:max_components]

        return psf_comp_t, psf_comp_s, h_det
    end

    # ----------------------------------------------------------------------
    # Function: simulate_lightsheet_image
    # Description: Simulates the light sheet image by convolving an object with the PSF components.
    # Arguments:
    #   - obj: The object array to simulate the light sheet on.
    #   - sz: The size of the output light sheet image.
    #   - psf_comp_t: Temporal PSF components from SVD decomposition.
    #   - psf_comp_s: Spatial PSF components from SVD decomposition.
    #   - h_det: The detection PSF.
    #   - fwd_components: The number of forward components to use in the simulation.
    # ----------------------------------------------------------------------
    function simulate_lightsheet_image(obj::AbstractArray{T, N}, sz, psf_comp_t, psf_comp_s, h_det, fwd_components) where {T, N}
        # Initialize the output image
        lightsheet_img = zeros(eltype(obj), sz)

        # Simulate the light sheet image by convolving the object with PSF components
        for n in 1:fwd_components
            # Combine temporal and detection PSF components
            psf_det_comb = reorient(psf_comp_t[n], Val(3)) .* h_det

            # Convolve the object with the spatial PSF and add to the image
            lightsheet_img += conv_psf(obj .* reorient(psf_comp_s[n], Val(1)), psf_det_comb)
        end

        return lightsheet_img
    end

end # module
