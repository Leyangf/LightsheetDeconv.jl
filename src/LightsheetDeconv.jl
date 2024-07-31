include("../src/LightsheetFwd.jl")

using .LightSheetSimulation
using InverseModeling
using SyntheticObjects, Noise, LinearAlgebra
using PointSpreadFunctions, FourierTools, Statistics
using View5D, GLMakie

# Define parameters
sz = (128, 128, 128)
pp_illu = PSFParams(0.5, 0.5, 1.52)
pp_det = PSFParams(0.5, 0.8, 1.52)
sampling = (0.20, 0.2, 0.200)
max_components = 4

# Define object
obj = filaments3D(sz)

# Simulate light-sheet image
psf_comp_x, psf_comp_y, h_det = LighSheetSimulation.simulate_lightsheet_psf(sz, pp_illu, pp_det, sampling, max_components)
lightsheet_img = LightSheetSimulation.simulate_lightsheet_image(obj, sz, psf_comp_x, psf_comp_y, h_det, max_components)

# Verify the lightsheet_img is generated correctly
@info "Simulated light-sheet image generated successfully."

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
nphotons = 100
nimg = poisson(pimg, nphotons)

# Optimize the model
start_val = (obj=Positive(mean(nimg) .* ones(Float64, size(nimg))),)
res1, myloss1 = optimize_model(start_val, forward_model, nimg)

# Visualize the results (optional)
@info "Optimization complete."
# @vt obj nimg res1[:obj]
