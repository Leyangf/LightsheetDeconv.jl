### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ ff4897c6-fef6-4071-bbbc-1c4fc204b990
# ╠═╡ disabled = true
#=╠═╡
begin
	using Pkg
	Pkg.activate("C://JuliaWork//LightsheetDeconv.jl")
	Pkg.instantiate()
end
  ╠═╡ =#

# ╔═╡ 2545dcb3-5a9f-414e-9659-3182c47278c5
using PlutoUI, PlutoTest

# ╔═╡ ebd51ba1-8a5a-4a9d-afba-d71825cc2195
using PointSpreadFunctions, LinearAlgebra, FourierTools, NDTools, SyntheticObjects, Statistics, Noise

# ╔═╡ 25337730-b2c3-4b79-9cfb-c9d709398334
using Plots

# ╔═╡ 45bc0270-d6c2-43c9-bdfc-c9ed2b2b165b
using GLMakie

# ╔═╡ d700b1bb-d20e-4ca1-8c83-10aa33fa23b3
using InverseModeling

# ╔═╡ bf5ef0ae-487f-4a8a-8933-4642eb2aca8b
using Printf

# ╔═╡ d32b6c5e-a724-4a32-b3d1-c0b802d028c5
md"# LightsheetDeconv.jl
	
In this notebook, we will work through the complete workflow, from simulating the illumination and detection PSFs of a light-sheet microscope to performing the deconvolution of an image captured by this setup.

The PSFs (Point Spread Functions) are modeled using **PointSpreadFunctions.jl**, which provides accurate representations of the optical blurring caused by the microscope. We will then apply **InverseModeling.jl** to deconvolve the image, enhancing the resolution by reversing the blurring effects introduced by the optical system.
"

# ╔═╡ 7784c364-95fa-4713-aaa3-f784bb12fd02
# Define the parameters to create an accurate model of the point spread function (PSF) based on your microscope's optical setup.

begin
	sz = (128, 128, 128) # dimensions of the 3D grid (in voxels) where the PSF will be calculated. 
	pp_illu = PSFParams(0.488, 0.33, 1.53) # parameters for the illumination PSF
	
	aberr = Aberrations([Zernike_Tilt, Zernike_Defocus, Zernike_VerticalAstigmatism, Zernike_VerticalTrefoil, Zernike_Spherical],[-0.78, -2.36, -1.3, 0.2, -1.2]) #
	pp_det = PSFParams(0.488, 0.56, 1.53; method=MethodPropagateIterative, aberrations= aberr) # parameters for the detection PSF
	
	sampling = (0.20, 0.20, 0.20) # sampling intervals in the x, y, and z directions, defines the voxel size for the grid
end

# ╔═╡ eb47b33e-078d-4e44-b621-e5d68555c54d
md"## 1. Simulate PSF"

# ╔═╡ 0e0b9ac7-3cb1-41d7-b90f-78bde0af8678
"""
	simulate_lightsheet_psf(sz, pp_illu, pp_det, sampling, max_components)

Simulate the 3D amplitude PSF with parameter `pp_illu`, collaspe along one dimention, and calculate the intensity PSF, which represents the side view of a light sheet.
Take the 2D Fourier transform and get the OTF. 
Perform SVD of OTF to decompose it into its principal components whcih are then stored in `otf_comp_t` and `otf_comp_s`.
Apply the inverse Fourier transform to each components to retrieve the PSF components in the spatial domain.
The 3D detection PSF `h_det` is also simulated with parameters `pp_det`.

The procedure of modelling illumination PSF is as following:
* generate 3d amplitutde PSF with `pp_illu` (use `apsf`)
* sum over the first dimension and take the squared magnitude
* sum over the fourth dimension which represents the electric field
* take the `fft` of it
* perform SVD and take main componnets of `F.U`, `F.S`, and `F.Vt`
* store the multiplication of `F.U` into `otf_comp_t` and 
* store the multiplication of `F.S` and `F.Vt` into `otf_comp_s`
* take the `ifft` of it
* store into `psf_comp_t` and `psf_comp_s`
"""
function simulate_lightsheet_psf(sz, pp_illu, pp_det, sampling, max_components)

    # illumination psf:
    h2d = sum(abs2.(sum(apsf(sz, pp_illu; sampling=sampling), dims=1)), dims=4)[1, :, :, 1]
    otf2d = fft(h2d)

    # detection psf:
    h_det = psf(sz, pp_det; sampling=sampling)

    F = svd(otf2d)

    otf_comp_s = Vector{Any}(undef, max_components)
    otf_comp_t = Vector{Any}(undef, max_components)
    otf_comp_s[1]= (F.S[1] * F.Vt[1, :])'
    otf_comp_t[1]= F.U[:, 1]
    for n in 2:max_components
        otf_comp_s[n]= (F.S[n] * F.Vt[n, :])'
        otf_comp_t[n]= F.U[:, n]
    end

    
    psf_comp_t = [real.(ifft(Array(otf_comp_t[n]), 1)) for n in 1:max_components]
    psf_comp_s = [real.(ifft(Array(otf_comp_s[n]), 2)) for n in 1:max_components]

    return psf_comp_t, psf_comp_s, h_det
end

# ╔═╡ 19c5aaad-7a8c-4d48-88ee-c3cdb81d591e
# simulate the first 20 PSF componets for strength and thickness and 3D PSF for detection
psf_comp_t, psf_comp_s, h_det = simulate_lightsheet_psf(sz, pp_illu, pp_det, sampling, 20)

# ╔═╡ e95ac99c-b330-4e5e-a82a-57bb57d5bb6a
@bind t1 PlutoUI.Slider(1:1:20, default=20, show_value=true)

# ╔═╡ 17aff1a1-2d8f-40ec-87cf-805bd657583e
# simulate reduced illumination PSF with first t1 componnets
psf_illu_red = sum(psf_comp_t[n] * psf_comp_s[n] for n in 1:t1)

# ╔═╡ c5765c7f-33ea-4e63-a1f8-5fa91c763c66
Plots.heatmap(psf_illu_red, title="Reduced Illumination PSF", aspect_ratio=:equal, xlabel="x",ylabel="z")

# ╔═╡ 7f1bf4d8-94aa-4344-9a18-29b702bcd5df
begin
	psf_illu_red_r = transpose(psf_illu_red)
	sheet_model_array = reshape(psf_illu_red_r, sz[1], 1, sz[2])
	sheet_model = repeat(sheet_model_array, 1, sz[3], 1)
	volume(sheet_model)
end

# ╔═╡ 160e41a3-70f2-4cd3-9208-c98fe336ea34
# generate original illumination PSF and compare with reduced one
begin
	expected_psf_illu = sum(abs2.(sum(apsf(sz, pp_illu; sampling=sampling), dims=1)), dims=4)[1, :, :, 1]
    dot_product = dot(psf_illu_red, expected_psf_illu)
    norm_product = norm(psf_illu_red) * norm(expected_psf_illu)
    ncc = dot_product / norm_product
PlutoTest.@test ncc ≈ 1.0 atol=1e-5
end

# ╔═╡ 0a9ad5a3-de51-4688-8f54-527b58e8eedf
@bind slice_index1 PlutoUI.Slider(1:128, default=64, show_value=true)

# ╔═╡ f8e4eb14-e710-47d1-b119-b5cd42a9275d
# plot each delected x-y slice of the 3d overall PSF, which is here simplized as a multiplication of illumination and detection PSF
begin
	psf_illu_red_20 = sum(reorient(psf_comp_t[n], Val(3)) .* reorient(psf_comp_s[n], Val(1)) for n in 1:20)
	psf_total_red = psf_illu_red_20 .* h_det
	Plots.heatmap(psf_total_red[:, :, slice_index1], title="Slice $slice_index1 of Reduced Overall PSF", aspect_ratio=:equal, xlabel="x",ylabel="y")
end

# ╔═╡ 61be4784-bbc6-436c-8d44-f590efaa3cfd
volume(psf_total_red)

# ╔═╡ 065c8a55-4e0a-4216-b98b-00fef317b036
md"## 2. Simulate Lightsheet image"

# ╔═╡ 71865013-0629-4794-b30a-0208e06766b4
"""
	simulate_lightsheet_image(obj::AbstractArray{T, N}, sz, psf_comp_t, psf_comp_s, h_det, fwd_components)

Simulate the generation of a blurred light-sheet microscopy image using the previously computed PSF components. 
Reorient the components onto x-z plane in the detection coordinate, propagating along x axis.
The multiplication of object with the x-axis PSF component represents a illuminated slice. The multiplication of detection PSF with the z-axis PSF component represents the total PSF along detection derection.
Convolve the two terms and get one image componnet, then stack along the fourth dimentsion.

The procedure is as following:
* initializes an empty image
* reorient the psf_comp_t and psf_comp_s from the illumination coordinate into detection coordinate
* multiply object with the x-axis PSF component
* multiply detection PSF with the z-axis PSF component
* convolve two terms
* loop iterates over each component of the PSF and computes the contribution of each to the final image
"""
function simulate_lightsheet_image(obj::AbstractArray{T, N}, sz, psf_comp_t, psf_comp_s, h_det, fwd_components) where {T, N}

    lightsheet_img = zeros(eltype(obj), sz)

    for n in 1:fwd_components
         psf_det_comb = reorient(psf_comp_t[n], Val(3)) .* h_det
         lightsheet_img += conv_psf(obj .* reorient(psf_comp_s[n], Val(1)), psf_det_comb)
         end
    return lightsheet_img
end

# ╔═╡ 0eba71d9-1520-4a8a-8168-9c39518253ce
function simulate_image_multip(obj::AbstractArray{T, N}, sz, psf_comp_t, psf_comp_s, h_det, fwd_components) where {T, N}

    lightsheet_img = zeros(eltype(obj), sz)

    for n in 1:fwd_components
         psf_comb = reorient(psf_comp_t[n], Val(3)).* reorient(psf_comp_s[n], Val(1)) .* h_det
         lightsheet_img += conv_psf(obj , psf_comb)
    end
    return lightsheet_img
end

# ╔═╡ a0abcdb1-93fa-420d-9081-ea4814a5cf7e
# Function to generate evenly spaced beads in 3D space with intensity
function create_evenly_beads(sz, num_beads::Int, bead_intensity::Float64)
    img = zeros(Float64, sz)
    # Calculate the step size for each dimension based on the cube root of the number of beads
    num_per_dim = ceil(Int, num_beads^(1/3))
    x_step = max(floor(Int, sz[1] / num_per_dim), 1)
    y_step = max(floor(Int, sz[2] / num_per_dim), 1)
    z_step = max(floor(Int, sz[3] / num_per_dim), 1)
    # Cartesian index generator with steps
    cartesian_indices = CartesianIndices((16:x_step:sz[1]-16, 16:y_step:sz[2]-16, 16:z_step:sz[3]-16))
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

# ╔═╡ 3a1cc28d-4b9f-4931-b60d-89f1b1016790
begin
# create evenly distributed beads and blur the beads image with 20 modes
beads_img = create_evenly_beads(sz, 500, 1.0)
beads_img_blur = simulate_lightsheet_image(beads_img, sz, psf_comp_t, psf_comp_s, h_det, 20)

volume(beads_img_blur)
end

# ╔═╡ 39fd0ef0-84c0-4605-9d6b-7281e4d2bb6d
begin
beads_blur_multip = simulate_image_multip(beads_img, sz, psf_comp_t, psf_comp_s, h_det, 20)

volume(beads_blur_multip)
end

# ╔═╡ a2edfbd5-c4a5-4e69-bcf6-cf5524c9ac85
begin
	# create two 3D images, one with a bead at center, while another one at corner
	
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
end

# ╔═╡ e25912f1-936e-4b02-ad26-d4cc9d684730
# Define bead positions: one in the corner and one in the center
begin
	corner_bead_position = (16, sz[2] ÷ 2, sz[3] ÷ 2)  # Near corner, but slightly offset to avoid out-of-bounds
	corner_bead_full_img = create_full_bead_image(sz, corner_bead_position, 1.5)
	corner_bead_full_blur = simulate_lightsheet_image(corner_bead_full_img, sz, psf_comp_t, psf_comp_s, h_det, 20)

	center_bead_position = (sz[1] ÷ 2, sz[2] ÷ 2, sz[3] ÷ 2)  # Center position
	center_bead_full_img = create_full_bead_image(sz, center_bead_position, 1.5)
	center_bead_full_blur = simulate_lightsheet_image(center_bead_full_img, sz, psf_comp_t, psf_comp_s, h_det, 20)
end

# ╔═╡ fb599d27-745f-4f11-be5c-a1d36821b8f3
sub_sz = (30, 30, 30) # Size of the small visualization region

# ╔═╡ 4c8f3a25-c1f4-4a82-9b52-350b2c68e6bd
begin
	center_bead_small_img = crop_subregion(center_bead_full_img, sub_sz, center_bead_position)
    center_bead_small_blur = crop_subregion(center_bead_full_blur, sub_sz, center_bead_position)
	volume(center_bead_small_blur)
end

# ╔═╡ 2a91dad5-7eec-4d6e-a99f-3f8762697596
begin
    # Crop the small regions around the bead positions
    corner_bead_small_img = crop_subregion(corner_bead_full_img, sub_sz, corner_bead_position)
    corner_bead_small_blur = crop_subregion(corner_bead_full_blur, sub_sz, corner_bead_position)
    volume(corner_bead_small_blur)
end

# ╔═╡ 7bbf50ea-8c0e-4643-a766-7d4ad99b941d
md"## 3. Perform Deconvolution"

# ╔═╡ 9efeaabe-2e09-4126-b940-b7433cbda4ce
"""
	perform_deconvolution(nimg, psf_comp_t, psf_comp_s, h_det, bwd_components)

Perform deconvolution on an image `nimg` using a forward model defined by the light-sheet simulation and the components of the point spread function (PSF). Using pakcage Inversemodeling.jl, the deconvolution optimizes the estimated object that, when convolved with the PSF, best matches the observed noisy image `nimg`.
"""
function perform_deconvolution(nimg, psf_comp_t, psf_comp_s, h_det, bwd_components)
    sz = size(nimg)
    # Define the forward model for deconvolution
    function forward_model(params)
        obj = params(:obj)
        return simulate_lightsheet_image(obj, sz, psf_comp_t, psf_comp_s, h_det, bwd_components)
    end

    # Create forward and backward models
    start_val = (obj=Positive(mean(nimg) .* ones(Float64, size(nimg))),)
    start_vals, fixed_vals, forward, backward, get_fit_results = create_forward(forward_model, start_val)

    # Optimize the model
    res, myloss = optimize_model(start_val, forward_model, nimg)

    return res
end

# ╔═╡ 356b8723-e38b-4b75-b33f-91fda1029451
@bind v1 PlutoUI.Slider(1:1:4, default=4, show_value=true)

# ╔═╡ 99d706fc-5a70-4def-9c14-6d3a93116ff1
# ╠═╡ disabled = true
#=╠═╡
# perform deconvolution of the beads image based on the first v1 componets of lightsheet PSF
begin
	res_beads  = perform_deconvolution(beads_img_blur, psf_comp_t, psf_comp_s, h_det, v1)
	beads_img_deconv = res_beads[:obj]
	volume(beads_img_deconv)
end
  ╠═╡ =#

# ╔═╡ 1e80f660-53d3-4464-9107-eae9ef8c92a3
# ╠═╡ disabled = true
#=╠═╡
# deconvolution result for the corner bead
begin
res_corner = perform_deconvolution(corner_bead_full_blur, psf_comp_t, psf_comp_s, h_det, v1)
corner_bead_small_deconv = crop_subregion(res_corner[:obj], sub_sz, corner_bead_position)
volume(corner_bead_small_deconv)
end
  ╠═╡ =#

# ╔═╡ e55e7639-9695-4ca2-87c5-362783e06859
# ╠═╡ disabled = true
#=╠═╡
# deconvolution result for the center bead
begin
res_center = perform_deconvolution(center_bead_full_blur, psf_comp_t, psf_comp_s, h_det, v1)
center_bead_small_deconv = crop_subregion(res_center[:obj], sub_sz, center_bead_position)
volume(center_bead_small_deconv)
end
  ╠═╡ =#

# ╔═╡ 077d00f5-b955-4f69-8846-ceda1c1accdf
md"# 4. Filaments Example"

# ╔═╡ bed1ff1b-c291-406e-b3c1-e87c9cd67caa
# create a 3D filaments image
fila_img = filaments3D(sz)

# ╔═╡ 6d1f2781-5c7b-4a1f-a6e0-9f4ea8d8d14b
volume(fila_img)

# ╔═╡ be1175a7-a315-482b-82b9-6d53bc927b75
@bind t2 PlutoUI.Slider(1:1:20, default=20, show_value=true)

# ╔═╡ 99a6e124-eac5-44f6-bce7-1b09f8599b2a
# blur the filaments image with the first t2 components of lightsheet PSF
begin
fila_img_blur = simulate_lightsheet_image(fila_img, sz, psf_comp_t, psf_comp_s, h_det, t2)
fila_nimg = poisson(fila_img_blur, 180)
volume(fila_nimg)
end

# ╔═╡ 6fa07b89-b9bd-48a2-9d39-04c6f3234de1
function compute_NCC(array1::Array{Float32, 3}, array2::Array{Float32, 3})
    # Ensure the arrays have the same dimensions
    if size(array1) != size(array2)
        throw(ArgumentError("Arrays must have the same dimensions"))
    end

    # Compute the means of the arrays
    mean1 = mean(array1)
    mean2 = mean(array2)

    # Subtract the means from the arrays (zero-mean arrays)
    array1_zero_mean = array1 .- mean1
    array2_zero_mean = array2 .- mean2

    # Compute the numerator: sum of element-wise product of zero-mean arrays
    numerator = sum(array1_zero_mean .* array2_zero_mean)

    # Compute the sum of squared zero-mean arrays (variance terms)
    sum_sq_array1 = sum(array1_zero_mean .^ 2)
    sum_sq_array2 = sum(array2_zero_mean .^ 2)

    # Compute the denominator: sqrt of the product of the variances
    denominator = sqrt(sum_sq_array1 * sum_sq_array2)

    # Check for zero denominator to avoid division by zero
    if denominator == 0
        throw(ArgumentError("Denominator is zero, which means one or both arrays have no variance."))
    end

    # Return the normalized cross-correlation
    NCC = numerator / denominator
    return NCC
end


# ╔═╡ 992cf406-72e6-43e1-a38b-2387cb778adb
@bind v2 PlutoUI.Slider(1:1:6, default=1, show_value=true)

# ╔═╡ d36af82b-a9ee-4c56-a67f-39907a70b39e
# ╠═╡ disabled = true
#=╠═╡
begin
# Measure the time taken for the deconvolution
elapsed_seconds = @elapsed begin
    # Perform deconvolution with v2 components
    res_fila = perform_deconvolution(fila_nimg, psf_comp_t, psf_comp_s, h_det, v2)
end

# Output the time taken for the deconvolution
@printf "Deconvolution Time = %.2f seconds\n" elapsed_seconds
end
  ╠═╡ =#

# ╔═╡ 333cbec1-af8c-43e9-930a-9e64071b82df
# ╠═╡ disabled = true
#=╠═╡
# perform deconvolution of the filaments image based on the first v2 componets of lightsheet PSF
begin
	fila_img_deconv = res_fila[:obj]
	volume(fila_img_deconv)
end
  ╠═╡ =#

# ╔═╡ fb9cc167-0227-41e0-ba2a-5d51b499285f
# ╠═╡ disabled = true
#=╠═╡
begin
# Compute NCC between the deconvolved image and the original image
ncc_value = compute_NCC(fila_img_deconv, fila_img)

# Output the NCC value
@printf "NCC Value = %.4f\n" ncc_value
end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╟─ff4897c6-fef6-4071-bbbc-1c4fc204b990
# ╠═2545dcb3-5a9f-414e-9659-3182c47278c5
# ╠═ebd51ba1-8a5a-4a9d-afba-d71825cc2195
# ╠═25337730-b2c3-4b79-9cfb-c9d709398334
# ╠═45bc0270-d6c2-43c9-bdfc-c9ed2b2b165b
# ╟─d32b6c5e-a724-4a32-b3d1-c0b802d028c5
# ╠═7784c364-95fa-4713-aaa3-f784bb12fd02
# ╟─eb47b33e-078d-4e44-b621-e5d68555c54d
# ╠═0e0b9ac7-3cb1-41d7-b90f-78bde0af8678
# ╠═19c5aaad-7a8c-4d48-88ee-c3cdb81d591e
# ╠═e95ac99c-b330-4e5e-a82a-57bb57d5bb6a
# ╠═17aff1a1-2d8f-40ec-87cf-805bd657583e
# ╠═c5765c7f-33ea-4e63-a1f8-5fa91c763c66
# ╠═7f1bf4d8-94aa-4344-9a18-29b702bcd5df
# ╠═160e41a3-70f2-4cd3-9208-c98fe336ea34
# ╠═0a9ad5a3-de51-4688-8f54-527b58e8eedf
# ╠═f8e4eb14-e710-47d1-b119-b5cd42a9275d
# ╠═61be4784-bbc6-436c-8d44-f590efaa3cfd
# ╠═065c8a55-4e0a-4216-b98b-00fef317b036
# ╟─71865013-0629-4794-b30a-0208e06766b4
# ╠═0eba71d9-1520-4a8a-8168-9c39518253ce
# ╠═a0abcdb1-93fa-420d-9081-ea4814a5cf7e
# ╠═3a1cc28d-4b9f-4931-b60d-89f1b1016790
# ╠═39fd0ef0-84c0-4605-9d6b-7281e4d2bb6d
# ╠═a2edfbd5-c4a5-4e69-bcf6-cf5524c9ac85
# ╠═e25912f1-936e-4b02-ad26-d4cc9d684730
# ╠═fb599d27-745f-4f11-be5c-a1d36821b8f3
# ╠═4c8f3a25-c1f4-4a82-9b52-350b2c68e6bd
# ╠═2a91dad5-7eec-4d6e-a99f-3f8762697596
# ╠═7bbf50ea-8c0e-4643-a766-7d4ad99b941d
# ╠═d700b1bb-d20e-4ca1-8c83-10aa33fa23b3
# ╠═9efeaabe-2e09-4126-b940-b7433cbda4ce
# ╠═356b8723-e38b-4b75-b33f-91fda1029451
# ╠═99d706fc-5a70-4def-9c14-6d3a93116ff1
# ╠═1e80f660-53d3-4464-9107-eae9ef8c92a3
# ╠═e55e7639-9695-4ca2-87c5-362783e06859
# ╠═077d00f5-b955-4f69-8846-ceda1c1accdf
# ╠═bed1ff1b-c291-406e-b3c1-e87c9cd67caa
# ╠═6d1f2781-5c7b-4a1f-a6e0-9f4ea8d8d14b
# ╠═be1175a7-a315-482b-82b9-6d53bc927b75
# ╠═99a6e124-eac5-44f6-bce7-1b09f8599b2a
# ╠═bf5ef0ae-487f-4a8a-8933-4642eb2aca8b
# ╠═6fa07b89-b9bd-48a2-9d39-04c6f3234de1
# ╠═992cf406-72e6-43e1-a38b-2387cb778adb
# ╠═d36af82b-a9ee-4c56-a67f-39907a70b39e
# ╠═333cbec1-af8c-43e9-930a-9e64071b82df
# ╠═fb9cc167-0227-41e0-ba2a-5d51b499285f