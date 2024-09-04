module LightSheetDeconv

export perform_deconvolution

include("LightsheetFwd.jl")

using .LightSheetSimulation
using InverseModeling
using PointSpreadFunctions, FourierTools, Statistics, LinearAlgebra

function perform_deconvolution(nimg, psf_comp_t, psf_comp_s, h_det, bwd_components)
    sz = size(nimg)
    # Define the forward model for deconvolution
    function forward_model(params)
        obj = params(:obj)
        return LightSheetSimulation.simulate_lightsheet_image(obj, sz, psf_comp_t, psf_comp_s, h_det, bwd_components)
    end

    # Create forward and backward models
    start_val = (obj=Positive(mean(nimg) .* ones(Float64, size(nimg))),)
    start_vals, fixed_vals, forward, backward, get_fit_results = create_forward(forward_model, start_val)

    # Optimize the model
    res, myloss = optimize_model(start_val, forward_model, nimg)

    return res
end

end # module
