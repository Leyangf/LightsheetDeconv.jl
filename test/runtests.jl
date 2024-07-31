include("../src/LightsheetDeconv.jl")

using .LightSheetDeconv, PointSpreadFunctions, SyntheticObjects, Noise, FourierTools
using Test
using View5D

# Define test parameters
sz = (128, 128, 128)
pp_illu = PSFParams(0.5, 0.5, 1.52)
pp_det = PSFParams(0.5, 0.8, 1.52)
sampling = (0.20, 0.2, 0.200)
max_components = 4
nphotons = 10

# Define object
obj = filaments3D(sz)

# Perform deconvolution
res1 = LightSheetDeconv.perform_deconvolution(obj, sz, pp_illu, pp_det, sampling, max_components, nphotons)

# Perform forward simulaiton
psf_comp_x, psf_comp_y, h_det = LightSheetSimulation.simulate_lightsheet_psf(sz, pp_illu, pp_det, sampling, max_components)
lightsheet_img = LightSheetSimulation.simulate_lightsheet_image(obj, sz, psf_comp_x, psf_comp_y, h_det, max_components)
nimg = poisson(lightsheet_img, nphotons)

@vt obj conv_psf(obj, h_det) nimg res1[:obj]

@test isapprox(sum(res1[:obj]), sum(obj), atol=1e-3)