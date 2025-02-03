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
sz = (64, 64, 64)
sampling = (0.13, 0.13, 0.13)
aberr = Aberrations([Zernike_Tilt, Zernike_ObliqueAstigmatism, Zernike_Defocus, Zernike_VerticalTrefoil, Zernike_HorizontalComa],[-0.78, -1.3, -2.36, -0.4, -1.21])
pp_det = PSFParams(0.525, 1, 1.35; method=MethodPropagateIterative, aberrations= aberr) # parameters for the detection PSF
pp_ill = PSFParams(0.488, 0.25, 1.35) # parameters for the illumination PSF

psf_comp_x, psf_comp_z = LightSheetSimulation.simulate_lightsheet_psf(sz, pp_ill, sampling, 4)
h_det = psf(sz, pp_det; sampling=sampling)

beads_img = TestFunctions.create_evenly_beads(sz, 1000, 3.0)
beads_img_blur = LightSheetSimulation.simulate_lightsheet_image(beads_img, sz, psf_comp_x, psf_comp_z, h_det, 4)
volume(beads_img_blur)
@vt beads_img_blur

# ----------------------------------------------------------------------
# Generate noisy image and evaluate the deconvolution result
# ----------------------------------------------------------------------
fila_img = filaments3D(sz)
fila_img_blur = LightSheetSimulation.simulate_lightsheet_image(fila_img, sz, psf_comp_x, psf_comp_z, h_det, 4)
fila_nimg = poisson(fila_img_blur, 1000)
volume(fila_nimg)

res_fila = LightSheetDeconv.perform_deconvolution(fila_nimg, psf_comp_x, psf_comp_z, h_det, 2; iterations=10, reg_weight=0.01, reg_type="goods")
fila_img_deconv = Float32.(res_fila[:obj])

ncc_value = TestFunctions.compute_NCC(fila_img_deconv, fila_img)
ssim_value = TestFunctions.ssim3d(fila_img_deconv, fila_img)

volume(fila_img_deconv)
@vt fila_img, fila_nimg, fila_img_deconv
