#include("../src/LightsheetFwd.jl")
include("../src/LightsheetDeconv.jl")

#using .LightSheetSimulation
using .LightSheetDeconv

using PointSpreadFunctions, FourierTools, NDTools, Noise
using SyntheticObjects
using View5D

# Define parameters
sz = (128, 128, 128)
pp_illu = PSFParams(0.5, 0.5, 1.52)
pp_det = PSFParams(0.5, 0.8, 1.52)
sampling = (0.20, 0.2, 0.200)

# Generate PSF components
psf_comp_x, psf_comp_y, h_det = LightSheetSimulation.simulate_lightsheet_psf(sz, pp_illu, pp_det, sampling, 20)


# Use beads image as object
function create_3d_beads_image(sz, num_beads::Int, bead_intensity::Float64)
    img = zeros(Float64, sz)
    for _ in 1:num_beads
        x, y, z = rand(1:sz[1]), rand(1:sz[2]), rand(1:sz[3])
        img[x, y, z] = bead_intensity
    end
    return img
end
beads_image = create_3d_beads_image(sz, 500, 2.0)
@vt beads_image

# Create lightsheet images with different componnets numbers
beads_img1 = LightSheetSimulation.simulate_lightsheet_image(beads_image, sz, psf_comp_x, psf_comp_y, h_det, 1)
beads_img4 = LightSheetSimulation.simulate_lightsheet_image(beads_image, sz, psf_comp_x, psf_comp_y, h_det, 4)
beads_img20 = LightSheetSimulation.simulate_lightsheet_image(beads_image, sz, psf_comp_x, psf_comp_y, h_det, 20)
@vt beads_image, lightsheet_img1, lightsheet_img4, lightsheet_img20

# nimg = poisson(lightsheet_img20, 10)

# Deconvolution of the beads image with main componnets
res1, myloss1 = LightSheetDeconv.perform_deconvolution(lightsheet_img20, psf_comp_x, psf_comp_y, h_det, 4)
@vt beads_image beads_img20 res1[:obj]


# Use filaments as object
filaments_image = filaments3D(sz)
# Create lightsheet image with 20 components
filaments_img20 = LightSheetSimulation.simulate_lightsheet_image(filaments_image, sz, psf_comp_x, psf_comp_y, h_det, 20)
# Deconvolution of the beads image with main componnets
res2, myloss2 = LightSheetDeconv.perform_deconvolution(lightsheet_img202, psf_comp_x, psf_comp_y, h_det, 1)
@vt obj lightsheet_img202 res2[:obj]