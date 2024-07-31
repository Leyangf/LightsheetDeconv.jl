module LightSheetDeconv

export simulate_lightsheet_psf, simulate_lightsheet_image, perform_deconvolution

include("LightsheetFwd.jl")

using .LightSheetSimulation
using InverseModeling
using SyntheticObjects, Noise, LinearAlgebra
using PointSpreadFunctions, FourierTools, Statistics
using View5D

function perform_deconvolution(obj, sz, pp_illu, pp_det, sampling, max_components, nphotons)
    # Simulate light-sheet image
    psf_comp_x, psf_comp_y, h_det = LightSheetSimulation.simulate_lightsheet_psf(sz, pp_illu, pp_det, sampling, max_components)
    # lightsheet_img = LightSheetSimulation.simulate_lightsheet_image(obj, sz, psf_comp_x, psf_comp_y, h_det, max_components)

    # Define the forward model for deconvolution
    function forward_model(params)
        obj = params(:obj)
        return LightSheetSimulation.simulate_lightsheet_image(obj, sz, psf_comp_x, psf_comp_y, h_det, max_components)
    end

    # Create forward and backward models
    start_val = (obj=obj,)
    start_vals, fixed_vals, forward, backward, get_fit_results = create_forward(forward_model, start_val)

    # Simulate and add noise
    pimg = forward(start_vals)
    nimg = poisson(pimg, nphotons)

    # Optimize the model
    start_val = (obj=Positive(mean(nimg) .* ones(Float64, size(nimg))),)
    res1, myloss1 = optimize_model(start_val, forward_model, nimg)

    return res1
end

end # module
