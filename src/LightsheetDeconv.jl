module LightSheetDeconv
    export perform_deconvolution

    # Include required dependencies and modules
    include("LightsheetFwd.jl")
    using .LightSheetSimulation
    using InverseModeling
    using PointSpreadFunctions, FourierTools, Statistics, LinearAlgebra

    # ----------------------------------------------------------------------
    # Function: perform_deconvolution
    # Description: Performs deconvolution on a noisy light sheet image using a forward model.
    # Arguments:
    #   - nimg: Noisy image data (e.g., blurred or degraded light sheet image).
    #   - psf_comp_t: Temporal PSF components from SVD decomposition.
    #   - psf_comp_s: Spatial PSF components from SVD decomposition.
    #   - h_det: The detection PSF.
    #   - bwd_components: The number of backward components to use in deconvolution.
    # ----------------------------------------------------------------------
    function perform_deconvolution(nimg, psf_comp_t, psf_comp_s, h_det, bwd_components)
        # Get the size of the noisy image
        sz = size(nimg)

        # Define the forward model for simulating the light sheet image during deconvolution
        function forward_model(params)
            obj = params(:obj)  # Extract object parameter from params
            return LightSheetSimulation.simulate_lightsheet_image(obj, sz, psf_comp_t, psf_comp_s, h_det, bwd_components)
        end

        # ----------------------------------------------------------------------
        # Set up initial parameters for deconvolution
        # Initial object guess: positive constraint with average intensity equal to mean(nimg)
        start_val = (obj=Positive(mean(nimg) .* ones(Float64, size(nimg))),)

        # Create forward and backward models using InverseModeling
        start_vals, fixed_vals, forward, backward, get_fit_results = create_forward(forward_model, start_val)

        # ----------------------------------------------------------------------
        # Optimize the model by minimizing the difference between forward model and noisy image (nimg)
        res, myloss = optimize_model(start_val, forward_model, nimg)

        # Return the result of the deconvolution (optimized object estimate)
        return res
    end

end # module
