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
begin
	using Pkg
	Pkg.activate("C://JuliaWork//LightsheetDeconv.jl")
	Pkg.instantiate()
end

# ╔═╡ 2545dcb3-5a9f-414e-9659-3182c47278c5
using PlutoUI

# ╔═╡ ebd51ba1-8a5a-4a9d-afba-d71825cc2195
using PointSpreadFunctions, LinearAlgebra, FourierTools, NDTools, SyntheticObjects, Statistics

# ╔═╡ 25337730-b2c3-4b79-9cfb-c9d709398334
using Plots

# ╔═╡ d700b1bb-d20e-4ca1-8c83-10aa33fa23b3
using InverseModeling

# ╔═╡ d32b6c5e-a724-4a32-b3d1-c0b802d028c5
md"# LightsheetDeconv.jl

In this homework we are going to work through the full pipeline from simulating the illumination PSF and detection PSF of a lightsheet microscope to deconvolution of an image captured by the setup."

# ╔═╡ 7784c364-95fa-4713-aaa3-f784bb12fd02
begin
	sz = (128, 128, 128)
	pp_illu = PSFParams(0.5, 0.5, 1.52)
	pp_det = PSFParams(0.5, 0.8, 1.52)
	sampling = (0.20, 0.2, 0.200)
end

# ╔═╡ eb47b33e-078d-4e44-b621-e5d68555c54d
md"## 1. Simulate PSF"

# ╔═╡ 0e0b9ac7-3cb1-41d7-b90f-78bde0af8678
"""
	simulate_PSF(s, r, [T=Float32])

Simulate the incoherent 2D PSF with radius `r` and total output size `s`.
This functions returns the PSF centered around the first entry (top left corner).
Furthermore, it returns the `freq_limit` of the PSF.
The `freq_limit` is the frequency at which the OTF becomes 0.

The procedure is as following:
* create a disc with values of 1 and radius `r` (use `rr2`)
* take the `ift` of it.
* abs2.
* normalize the sum to 1
* center pixcel should be located in the top left
"""

# ╔═╡ 5d83458a-e7fd-4299-9fe4-11247c4413a4
function simulate_lightsheet_psf(sz, pp_illu, pp_det, sampling, max_components)

    # illumination psf:
    h2d = sum(abs2.(sum(apsf(sz, pp_illu; sampling=sampling), dims=1)), dims=4)[1, :, :, 1]
    otf2d = fft(h2d)

    # detection psf:
    h_det = psf(sz, pp_det; sampling=sampling)

    F = svd(otf2d)

    otf_comp_x = Vector{Any}(undef, max_components)
    otf_comp_y = Vector{Any}(undef, max_components)
    otf_comp_y[1]= (F.S[1] * F.Vt[1, :])'
    otf_comp_x[1]= F.U[:, 1]
    for n in 2:max_components
        otf_comp_y[n]= (F.S[n] * F.Vt[n, :])'
        otf_comp_x[n]= F.U[:, n]
    end

    # iFT to get the PSF componnets
    psf_comp_x = [real.(ifft(Array(otf_comp_x[n]), 1)) for n in 1:max_components]
    psf_comp_y = [real.(ifft(Array(otf_comp_y[n]), 2)) for n in 1:max_components]

    return psf_comp_x, psf_comp_y, h_det
end

# ╔═╡ 19c5aaad-7a8c-4d48-88ee-c3cdb81d591e
psf_comp_x, psf_comp_y, h_det = simulate_lightsheet_psf(sz, pp_illu, pp_det, sampling, 20)

# ╔═╡ e95ac99c-b330-4e5e-a82a-57bb57d5bb6a
@bind t1 PlutoUI.Slider(1:1:20, default=20, show_value=true)

# ╔═╡ c5765c7f-33ea-4e63-a1f8-5fa91c763c66
begin
	psf_illu_red = sum(psf_comp_x[n] * psf_comp_y[n] for n in 1:t1)
	Plots.heatmap(psf_illu_red, title="Reduced Illumination PSF", aspect_ratio=:equal)
end

# ╔═╡ 0a9ad5a3-de51-4688-8f54-527b58e8eedf
@bind slice_index1 PlutoUI.Slider(1:128, default=64, show_value=true)

# ╔═╡ f8e4eb14-e710-47d1-b119-b5cd42a9275d
begin
	psf_total_red = psf_illu_red .* h_det
	Plots.heatmap(psf_total_red[:, :, slice_index1], title="Slice $slice_index1 of Reduced Overall PSF", aspect_ratio=:equal)
end

# ╔═╡ 065c8a55-4e0a-4216-b98b-00fef317b036
md"## 3. Simulate Lightsheet image"

# ╔═╡ 71865013-0629-4794-b30a-0208e06766b4
function simulate_lightsheet_image(obj::AbstractArray{T, N}, sz, psf_comp_x, psf_comp_y, h_det, fwd_components) where {T, N}

    lightsheet_img = zeros(eltype(obj), sz)
    for n in 1:fwd_components
        # psf_total_comp = reorient(psf_comp_x[n], Val(3)) .* h_det
        lightsheet_img += conv_psf(obj .* reorient(psf_comp_y[n], Val(1)), reorient(psf_comp_x[n], Val(3)) .* h_det)
    end
    return lightsheet_img
end

# ╔═╡ a2edfbd5-c4a5-4e69-bcf6-cf5524c9ac85
function create_3d_beads_image(sz, num_beads::Int, bead_intensity::Float64)
    img = zeros(Float64, sz)
    for _ in 1:num_beads
        x, y, z = rand(1:sz[1]), rand(1:sz[2]), rand(1:sz[3])
        img[x, y, z] = bead_intensity
    end
    return img
end

# ╔═╡ 41624095-2399-4867-b8fb-c49d424231ba
@bind slice_index2 PlutoUI.Slider(1:128, default=64, show_value=true)

# ╔═╡ 0e0dcb3b-21fd-416d-a8e3-92d38c5e5e10
begin
	beads_img = create_3d_beads_image(sz, 500, 2.0)
	beads_img_blur = simulate_lightsheet_image(beads_img, sz, psf_comp_x, psf_comp_y, h_det, t1)
	
	Plots.heatmap(beads_img_blur[:, :, slice_index2], title="Slice $slice_index2 of blurred beads image", aspect_ratio=:equal)
end

# ╔═╡ 7bbf50ea-8c0e-4643-a766-7d4ad99b941d
md"## 3. Perform Deconvolution"

# ╔═╡ 9efeaabe-2e09-4126-b940-b7433cbda4ce
function perform_deconvolution(nimg, psf_comp_x, psf_comp_y, h_det, bwd_components)
    sz = size(nimg)
    # Define the forward model for deconvolution
    function forward_model(params)
        obj = params(:obj)
        return simulate_lightsheet_image(obj, sz, psf_comp_x, psf_comp_y, h_det, bwd_components)
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

# ╔═╡ 1e80f660-53d3-4464-9107-eae9ef8c92a3
res1 = perform_deconvolution(beads_img_blur, psf_comp_x, psf_comp_y, h_det, v1)
    beads_img_deconv = res1[:obj]

# ╔═╡ 1dae517b-d1ec-4302-9013-9caa802f73c5
@bind slice_index3 PlutoUI.Slider(1:128, default=64, show_value=true)

# ╔═╡ baeafd3d-380e-4561-9792-7f60d4e19bc1
plot(heatmap(beads_img[:, :, slice_index3], title="Slice $slice_index3 of original beads image", aspect_ratio=:equal), heatmap(beads_img_deconv[:, :, slice_index3], title="Slice $slice_index3 of deconvolved beads image", aspect_ratio=:equal))

# ╔═╡ 077d00f5-b955-4f69-8846-ceda1c1accdf
md"# 4. Real Example"

# ╔═╡ be1175a7-a315-482b-82b9-6d53bc927b75
@bind t2 PlutoUI.Slider(1:1:20, default=20, show_value=true)

# ╔═╡ 99a6e124-eac5-44f6-bce7-1b09f8599b2a
begin
	fila_img = filaments3D(sz)
	
	fila_img_blur = simulate_lightsheet_image(fila_img, sz, psf_comp_x, psf_comp_y, h_det, t2)
end

# ╔═╡ 992cf406-72e6-43e1-a38b-2387cb778adb
@bind v2 PlutoUI.Slider(1:1:4, default=4, show_value=true)

# ╔═╡ 0cdce5f8-8f95-457d-b493-61b90e218b64
res2 = perform_deconvolution(fila_img_blur, psf_comp_x, psf_comp_y, h_det, v2)
	fila_img_deconv = res1[:obj]

# ╔═╡ 305713f1-ef89-4ed2-a2de-df56b5753dda
@bind slice_index4 PlutoUI.Slider(1:128, default=64, show_value=true)

# ╔═╡ 380271ab-9b8b-46cf-8ffb-ce232c75e76b
#=╠═╡
plot(heatmap(fila_img[:, :, slice_index4], title="Slice $slice_index4 of original filaments image", aspect_ratio=:equal), heatmap(fila_img_blur[:, :, slice_index4], title="Slice $slice_index4 of blurred filaments image", aspect_ratio=:equal), heatmap(fila_img_deconv[:, :, slice_index4], title="Slice $slice_index4 of deconvolved filaments image", aspect_ratio=:equal))
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═ff4897c6-fef6-4071-bbbc-1c4fc204b990
# ╠═d32b6c5e-a724-4a32-b3d1-c0b802d028c5
# ╠═2545dcb3-5a9f-414e-9659-3182c47278c5
# ╠═ebd51ba1-8a5a-4a9d-afba-d71825cc2195
# ╠═25337730-b2c3-4b79-9cfb-c9d709398334
# ╠═7784c364-95fa-4713-aaa3-f784bb12fd02
# ╠═eb47b33e-078d-4e44-b621-e5d68555c54d
# ╠═0e0b9ac7-3cb1-41d7-b90f-78bde0af8678
# ╠═5d83458a-e7fd-4299-9fe4-11247c4413a4
# ╠═19c5aaad-7a8c-4d48-88ee-c3cdb81d591e
# ╠═e95ac99c-b330-4e5e-a82a-57bb57d5bb6a
# ╠═c5765c7f-33ea-4e63-a1f8-5fa91c763c66
# ╠═0a9ad5a3-de51-4688-8f54-527b58e8eedf
# ╠═f8e4eb14-e710-47d1-b119-b5cd42a9275d
# ╠═065c8a55-4e0a-4216-b98b-00fef317b036
# ╠═71865013-0629-4794-b30a-0208e06766b4
# ╠═a2edfbd5-c4a5-4e69-bcf6-cf5524c9ac85
# ╠═41624095-2399-4867-b8fb-c49d424231ba
# ╠═0e0dcb3b-21fd-416d-a8e3-92d38c5e5e10
# ╠═7bbf50ea-8c0e-4643-a766-7d4ad99b941d
# ╠═d700b1bb-d20e-4ca1-8c83-10aa33fa23b3
# ╠═9efeaabe-2e09-4126-b940-b7433cbda4ce
# ╠═356b8723-e38b-4b75-b33f-91fda1029451
# ╠═1e80f660-53d3-4464-9107-eae9ef8c92a3
# ╠═1dae517b-d1ec-4302-9013-9caa802f73c5
# ╠═baeafd3d-380e-4561-9792-7f60d4e19bc1
# ╠═077d00f5-b955-4f69-8846-ceda1c1accdf
# ╠═be1175a7-a315-482b-82b9-6d53bc927b75
# ╠═99a6e124-eac5-44f6-bce7-1b09f8599b2a
# ╠═992cf406-72e6-43e1-a38b-2387cb778adb
# ╠═0cdce5f8-8f95-457d-b493-61b90e218b64
# ╠═305713f1-ef89-4ed2-a2de-df56b5753dda
# ╠═380271ab-9b8b-46cf-8ffb-ce232c75e76b