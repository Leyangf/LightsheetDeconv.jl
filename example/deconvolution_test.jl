# ----------------------------------------------------------------------
# Importing necessary libraries and setting up configurations
# ----------------------------------------------------------------------

include("../src/LightsheetDeconv.jl")
include("../src/util.jl")

using .LightSheetDeconv
using .TestFunctions

using PointSpreadFunctions, FourierTools, NDTools, Noise
using SyntheticObjects
using View5D
using GLMakie

# ----------------------------------------------------------------------
# Define the Point Spread Function (PSF)
# ----------------------------------------------------------------------

# Define parameters for PSF simulation
sz = (128, 128, 128)
pp_illu = PSFParams(0.525, 0.2, 1.53)
aberr = Aberrations(
    [Zernike_Tilt, Zernike_Defocus, Zernike_VerticalAstigmatism, Zernike_VerticalTrefoil, Zernike_Spherical],
    [-0.78, -1.36, -1.3, 0.2, -1.2]
)
pp_det = PSFParams(0.525, 0.5, 1.53; method=MethodPropagateIterative, aberrations=aberr)
sampling = (0.20, 0.20, 0.20)

# Generate PSF components
psf_comp_t, psf_comp_s, h_det = LightSheetSimulation.simulate_lightsheet_psf(sz, pp_illu, pp_det, sampling, 20)

# ----------------------------------------------------------------------
# Function to load and blur the image
# ----------------------------------------------------------------------

# Define bead positions
corner_bead_position = (16, sz[2] ÷ 2, sz[3] ÷ 2)  # Near corner, offset to avoid out-of-bounds
center_bead_position = (sz[1] ÷ 2, sz[2] ÷ 2, sz[3] ÷ 2)  # Center position

# Create full-size images with the beads
corner_bead_full_img = TestImages.create_full_bead_image(sz, corner_bead_position, 2.0)
center_bead_full_img = TestImages.create_full_bead_image(sz, center_bead_position, 2.0)

# Simulate light-sheet images for the full-size regions
corner_bead_full_blur = LightSheetSimulation.simulate_lightsheet_image(corner_bead_full_img, sz, psf_comp_t, psf_comp_s, h_det, 20)
center_bead_full_blur = LightSheetSimulation.simulate_lightsheet_image(center_bead_full_img, sz, psf_comp_t, psf_comp_s, h_det, 20)

# Define the size of the full image and the smaller sub-region
sub_sz = (30, 30, 30)

# Crop the small regions around the bead positions
corner_bead_small_img = TestImages.crop_subregion(corner_bead_full_img, sub_sz, corner_bead_position)
corner_bead_small_blur = TestImages.crop_subregion(corner_bead_full_blur, sub_sz, corner_bead_position)
center_bead_small_img = TestImages.crop_subregion(center_bead_full_img, sub_sz, center_bead_position)
center_bead_small_blur = TestImages.crop_subregion(center_bead_full_blur, sub_sz, center_bead_position)    

# Visualize the small regions for comparison
volume(corner_bead_small_blur)
volume(center_bead_small_blur)
@vt corner_bead_small_blur, center_bead_small_blur


# Generate and blur evenly spaced beads image
beads_img = TestImages.create_evenly_beads(sz, 500, 3.0)
beads_img_blur = LightSheetSimulation.simulate_lightsheet_image(beads_img, sz, psf_comp_t, psf_comp_s, h_det, 20)
volume(beads_img_blur)
@vt beads_img_blur


# Use filaments as object
fila_img = filaments3D(sz)
fila_img_blur = LightSheetSimulation.simulate_lightsheet_image(fila_img, sz, psf_comp_t, psf_comp_s, h_det, 20)
fila_nimg = poisson(fila_img_blur, 180)
volume(fila_nimg)

# ----------------------------------------------------------------------
# Deconvolution of the image with main componnets
# ----------------------------------------------------------------------

# Perform deconvolution for corner and center bead images
res_corner = LightSheetDeconv.perform_deconvolution(corner_bead_full_blur, psf_comp_t, psf_comp_s, h_det, 4)
res_center = LightSheetDeconv.perform_deconvolution(center_bead_full_blur, psf_comp_t, psf_comp_s, h_det, 4)

# Crop the deconvolved results
corner_bead_small_deconv = TestImages.crop_subregion(res_corner[:obj], sub_sz, corner_bead_position)
center_bead_small_deconv = TestImages.crop_subregion(res_center[:obj], sub_sz, center_bead_position)

# Visualize the deconvolved images
volume(corner_bead_small_deconv)
volume(center_bead_small_deconv)
@vt corner_bead_small_img, corner_bead_small_blur, corner_bead_small_deconv

# Deconvolution of the beads image
res_beads = LightSheetDeconv.perform_deconvolution(beads_img_blur, psf_comp_t, psf_comp_s, h_det, 4)
beads_img_deconv = res_beads[:obj]
volume(beads_img_deconv)

# Deconvolution of the filaments image
res_fila = LightSheetDeconv.perform_deconvolution(fila_nimg, psf_comp_t, psf_comp_s, h_det, 4)
fila_img_deconv = res_fila[:obj]
volume(fila_img_deconv)
@vt fila_img, fila_nimg, fila_img_deconv

# Compute and display NCC (Normalized Cross-Correlation) between original and deconvolved filament images
ncc_value = TestImages.compute_NCC(fila_img_deconv, fila_img)