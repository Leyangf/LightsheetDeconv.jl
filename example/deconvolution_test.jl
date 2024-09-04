# include("../src/LightsheetFwd.jl")
include("../src/LightsheetDeconv.jl")

#using .LightSheetSimulation
using .LightSheetDeconv
# using .LightSheetSimulation

using PointSpreadFunctions, FourierTools, NDTools, Noise
using SyntheticObjects
using View5D

# Define parameters
sz = (256, 256, 256)
pp_illu = PSFParams(0.488, 0.15, 1.53)
aberr = Aberrations([Zernike_Tilt, Zernike_Defocus, Zernike_VerticalAstigmatism, Zernike_VerticalTrefoil, Zernike_Spherical],[-0.78, -2.36, -1.3, 0.2, -1.2])
pp_det = PSFParams(0.488, 0.56, 1.53; method=MethodPropagateIterative, aberrations= aberr)
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

# Define the size of the full image and the smaller sub-region
sub_sz = (60, 60, 60)

# Define bead positions: one in the corner and one in the center
corner_bead_position = (50, 50, 50)  # Near corner, but slightly offset to avoid out-of-bounds
center_bead_position = (sz[1] ÷ 2, sz[2] ÷ 2, sz[3] ÷ 2)  # Center position

# Create full-size images with the beads
corner_bead_full_img = create_full_bead_image(sz, corner_bead_position, 3.0)
center_bead_full_img = create_full_bead_image(sz, center_bead_position, 3.0)

# Simulate the light-sheet images for the full-size regions
corner_bead_full_blur = LightSheetSimulation.simulate_lightsheet_image(corner_bead_full_img, sz, psf_comp_t, psf_comp_s, h_det, 20)
center_bead_full_blur = LightSheetSimulation.simulate_lightsheet_image(center_bead_full_img, sz, psf_comp_t, psf_comp_s, h_det, 20)

# Crop the small regions around the bead positions
corner_bead_small_img = crop_subregion(corner_bead_full_img, sub_sz, corner_bead_position)
corner_bead_small_blur = crop_subregion(corner_bead_full_blur, sub_sz, corner_bead_position)

center_bead_small_img = crop_subregion(center_bead_full_img, sub_sz, center_bead_position)
center_bead_small_blur = crop_subregion(center_bead_full_blur, sub_sz, center_bead_position)

# Visualize the small regions for comparison
@vt corner_bead_small_blur
@vt center_bead_small_blur



# Use beads image as object
function create_3d_beads_image(sz, num_beads::Int, bead_intensity::Float64)
    img = zeros(Float64, sz)
    for _ in 1:num_beads
        x, y, z = rand(1:sz[1]), rand(1:sz[2]), rand(1:sz[3])
        img[x, y, z] = bead_intensity
    end
    return img
end
beads_img = create_3d_beads_image(sz, 500, 5.0)

# Create lightsheet images with different componnets numbers
beads_img_blur = LightSheetSimulation.simulate_lightsheet_image(beads_img, sz, psf_comp_t, psf_comp_s, h_det, 20)
@vt beads_img, beads_img_blur



# nimg = poisson(lightsheet_img20, 10)

# Deconvolution of the beads image with main componnets
res1 = LightSheetDeconv.perform_deconvolution(beads_img_blur, psf_comp_y, psf_comp_z, h_det, 4)
@vt beads_img beads_img_blur res1[:obj]


# Use filaments as object
fila_img = filaments3D(sz)
# Create lightsheet image with 20 components
fila_img_blur = LightSheetSimulation.simulate_lightsheet_image(fila_img, sz, psf_comp_y, psf_comp_z, h_det, 20)
# Deconvolution of the beads image with main componnets
res2  = LightSheetDeconv.perform_deconvolution(fila_img_blur, psf_comp_y, psf_comp_z, h_det, 4)
@vt fila_img, fila_img_blur res2[:obj]