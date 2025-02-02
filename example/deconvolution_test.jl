include("../src/LightsheetDeconv.jl")
include("../src/util.jl")

using .LightSheetDeconv
using .TestFunctions

using PointSpreadFunctions, FourierTools, NDTools, Noise
using SyntheticObjects
using View5D
using GLMakie

# ----------------------------------------------------------------------
# Define PSF
# ----------------------------------------------------------------------
sz = (128, 128, 128)
sampling = (0.13, 0.13, 0.13)
aberr = Aberrations([Zernike_Tilt, Zernike_ObliqueAstigmatism, Zernike_Defocus, Zernike_VerticalTrefoil, Zernike_HorizontalComa],[-0.78, -1.3, -2.36, -0.4, -1.21])
pp_det = PSFParams(0.525, 1, 1.35; method=MethodPropagateIterative, aberrations= aberr) # parameters for the detection PSF
pp_ill = PSFParams(0.488, 0.25, 1.35) # parameters for the illumination PSF

psf_comp_x, psf_comp_z = LightSheetSimulation.simulate_lightsheet_psf(sz, pp_ill, sampling, 4)
h_det = psf(sz, pp_det; sampling=sampling)

# ----------------------------------------------------------------------
# Generate noisy images
# ----------------------------------------------------------------------

beads_img = TestFunctions.create_evenly_beads(sz, 1000, 3.0)
beads_img_blur = LightSheetSimulation.simulate_lightsheet_image(beads_img, sz, psf_comp_x, psf_comp_z, h_det, 2)
volume(beads_img_blur)
@vt beads_img_blur

fila_img = filaments3D(sz)
fila_img_blur = LightSheetSimulation.simulate_lightsheet_image(fila_img, sz, psf_comp_x, psf_comp_z, h_det, 2)
fila_nimg = poisson(fila_img_blur, 180)
volume(fila_nimg)

# ----------------------------------------------------------------------
# Deconvolution of the image with main componnets
# ----------------------------------------------------------------------

# Deconvolution of the filaments image
res_fila = LightSheetDeconv.perform_deconvolution(fila_nimg, psf_comp_x, psf_comp_z, h_det, 2; iterations=20, goods_weight=0.01)
fila_img_deconv = res_fila[:obj]
volume(fila_img_deconv)
@vt fila_img, fila_nimg, fila_img_deconv

# Compute and display NCC (Normalized Cross-Correlation) between original and deconvolved filament images
ncc_value = TestFunctions.compute_NCC(fila_img_deconv, fila_img)