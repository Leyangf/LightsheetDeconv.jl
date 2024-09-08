# include("../src/LightsheetFwd.jl")
include("../src/LightsheetDeconv.jl")

#using .LightSheetSimulation
using .LightSheetDeconv
# using .LightSheetSimulation

using PointSpreadFunctions, FourierTools, NDTools, Noise
using SyntheticObjects
using View5D
using GLMakie

# Define parameters
sz = (128, 128, 128)
pp_illu = PSFParams(0.488, 0.2, 1.53)
aberr = Aberrations([Zernike_Tilt, Zernike_Defocus, Zernike_VerticalAstigmatism, Zernike_VerticalTrefoil, Zernike_Spherical],[-0.78, -1.36, -1.3, 0.2, -1.2])
pp_det = PSFParams(0.488, 0.5, 1.53; method=MethodPropagateIterative, aberrations= aberr)
sampling = (0.20, 0.20, 0.20)

# Generate PSF components
psf_comp_t, psf_comp_s, h_det = LightSheetSimulation.simulate_lightsheet_psf(sz, pp_illu, pp_det, sampling, 20)


    function create_full_bead_image(sz, bead_position::Tuple{Int,Int,Int}, bead_intensity::Float64)
        img = zeros(Float64, sz)
        x, y, z = bead_position
        img[x, y, z] = bead_intensity
        return img
    end
    
    function crop_subregion(img, sub_sz, center_pos)
        # Calculate the start and end indices for cropping
        start_idx = (center_pos[1] - sub_sz[1] ÷ 2,
                     center_pos[2] - sub_sz[2] ÷ 2,
                     center_pos[3] - sub_sz[3] ÷ 2)
        end_idx = (start_idx[1] + sub_sz[1] - 1,
                   start_idx[2] + sub_sz[2] - 1,
                   start_idx[3] + sub_sz[3] - 1)
        
        # Ensure indices are within bounds
        start_idx = clamp.(start_idx, 1, size(img))
        end_idx = clamp.(end_idx, 1, size(img))
        
        # Crop the region
        return img[start_idx[1]:end_idx[1], start_idx[2]:end_idx[2], start_idx[3]:end_idx[3]]
    end
    
    # Define bead positions: one in the corner and one in the center
    corner_bead_position = (16, sz[2] ÷ 2, sz[3] ÷ 2)  # Near corner, but slightly offset to avoid out-of-bounds
    center_bead_position = (sz[1] ÷ 2, sz[2] ÷ 2, sz[3] ÷ 2)  # Center position
    
    # Create full-size images with the beads
    corner_bead_full_img = create_full_bead_image(sz, corner_bead_position, 2.0)
    center_bead_full_img = create_full_bead_image(sz, center_bead_position, 2.0)
    
    # Simulate the light-sheet images for the full-size regions
    corner_bead_full_blur = LightSheetSimulation.simulate_lightsheet_image(corner_bead_full_img, sz, psf_comp_t, psf_comp_s, h_det, 20)
    center_bead_full_blur = LightSheetSimulation.simulate_lightsheet_image(center_bead_full_img, sz, psf_comp_t, psf_comp_s, h_det, 20)
    
    # Define the size of the full image and the smaller sub-region
    sub_sz = (30, 30, 30)

    # Crop the small regions around the bead positions
    corner_bead_small_img = crop_subregion(corner_bead_full_img, sub_sz, corner_bead_position)
    corner_bead_small_blur = crop_subregion(corner_bead_full_blur, sub_sz, corner_bead_position)
    
    center_bead_small_img = crop_subregion(center_bead_full_img, sub_sz, center_bead_position)
    center_bead_small_blur = crop_subregion(center_bead_full_blur, sub_sz, center_bead_position)
    
    # Visualize the small regions for comparison
    volume(corner_bead_small_blur)
    volume(center_bead_small_blur)

    @vtp corner_bead_small_blur, center_bead_small_blur


# Deconvolution of the beads image with main componnets
res_corner = LightSheetDeconv.perform_deconvolution(corner_bead_full_blur, psf_comp_t, psf_comp_s, h_det, 4)
res_center = LightSheetDeconv.perform_deconvolution(center_bead_full_blur, psf_comp_t, psf_comp_s, h_det, 4)
corner_bead_small_deconv = crop_subregion(res_corner[:obj], sub_sz, corner_bead_position)
center_bead_small_deconv = crop_subregion(res_center[:obj], sub_sz, center_bead_position)

volume(corner_bead_small_deconv)
volume(center_bead_small_deconv)

@vt corner_bead_small_img corner_bead_small_blur corner_bead_small_deconv


# Use beads image as object
# Function to generate evenly spaced beads in 3D space with intensity
function create_evenly_beads(sz, num_beads::Int, bead_intensity::Float64)
    img = zeros(Float64, sz)
    # Calculate the step size for each dimension based on the cube root of the number of beads
    num_per_dim = ceil(Int, num_beads^(1/3))
    x_step = max(floor(Int, sz[1] / num_per_dim), 1)
    y_step = max(floor(Int, sz[2] / num_per_dim), 1)
    z_step = max(floor(Int, sz[3] / num_per_dim), 1)
    # Cartesian index generator with steps
    cartesian_indices = CartesianIndices((20:x_step:sz[1]-20, 20:y_step:sz[2]-20, 20:z_step:sz[3]-20))
    # Add beads to the image based on the Cartesian indices
    for idx in cartesian_indices
        if num_beads > 0
            img[idx] = bead_intensity
            num_beads -= 1
        else
            return img
        end
    end
    return img
end

# Generate blurred beads image
    beads_img = create_evenly_beads(sz, 500, 3.0)
    beads_img_blur = LightSheetSimulation.simulate_lightsheet_image(beads_img, sz, psf_comp_t, psf_comp_s, h_det, 20)
    
    volume(beads_img_blur)
    @vt beads_img_blur


# Use filaments as object
fila_img = filaments3D(sz)
# Create lightsheet image with 20 components
fila_img_blur = LightSheetSimulation.simulate_lightsheet_image(fila_img, sz, psf_comp_t, psf_comp_s, h_det, 20)
nimg = poisson(fila_img_blur, 80)
volume(fila_nimg)

# Deconvolution of the beads image with main componnets
res_fila  = LightSheetDeconv.perform_deconvolution(fila_nimg, psf_comp_t, psf_comp_s, h_det, 4)
volume(res_fila[:obj])
@vt fila_img, fila_img_blur res_fila[:obj]