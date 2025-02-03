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

# ----------------------------------------------------------------------
# Generate noisy image and evaluate the deconvolution result
# ----------------------------------------------------------------------
# Blurred image
beads_img = Float32.(TestFunctions.create_evenly_beads(sz, 1000, 3.0))
beads_img_blur = LightSheetSimulation.simulate_lightsheet_image(beads_img, sz, psf_comp_x, psf_comp_z, h_det, 4)

fila_img = filaments3D(sz)
fila_img_blur = LightSheetSimulation.simulate_lightsheet_image(fila_img, sz, psf_comp_x, psf_comp_z, h_det, 4)


# Optimization
img = fila_img
nimg = poisson(fila_img_blur, 1000)

# non-regularizer deconvolution result
res0 = LightSheetDeconv.perform_deconvolution(nimg, psf_comp_x, psf_comp_z, h_det, 2; iterations=120, reg_weight=0, reg_type="tv")
img_deconv0 = Float32.(res0[:obj])
ncc_value0 = TestFunctions.compute_NCC(img_deconv0, img)

# with regualrizer
res1 = LightSheetDeconv.perform_deconvolution(nimg, psf_comp_x, psf_comp_z, h_det, 2; iterations=33, reg_weight=0.003, reg_type="tv")
img_deconv1 = Float32.(res1[:obj])
ncc_value1 = TestFunctions.compute_NCC(img_deconv1, img)


# other compare method
ssim_value0 = TestFunctions.ssim3d(img_deconv0, img)
ssim_value1 = TestFunctions.ssim3d(img_deconv1, img)
volume(img_deconv0)
@vt img, nimg, img_deconv1


using Base.Threads
iterations_range = 25:38
reg_weight_range = 0.0008:0.0002:0.03

best_ncc = -Inf
best_iterations = 0
best_reg_weight = 0.0

for iterations in iterations_range
    for reg_weight in reg_weight_range
        try
            res = LightSheetDeconv.perform_deconvolution(nimg, psf_comp_x, psf_comp_z, h_det, 2;
                iterations=iterations, reg_weight=reg_weight, reg_type="tv")
            img_deconv = Float32.(res[:obj])
            
            ncc_value = TestFunctions.compute_NCC(img_deconv, img)
            
            if ncc_value > best_ncc
                best_ncc = ncc_value
                best_iterations = iterations
                best_reg_weight = reg_weight
            end
            
            println("iterations=$iterations, reg_weight=$reg_weight, NCC=$ncc_value")
        catch e
            println("Error at iterations=$iterations, reg_weight=$reg_weight: $e")
        end
    end
end

println("Best combination: iterations=$best_iterations, reg_weight=$best_reg_weight, NCC=$best_ncc")
