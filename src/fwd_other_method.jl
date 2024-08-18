using PointSpreadFunctions, View5D, LinearAlgebra, FFTW, NDTools, SyntheticObjects
# using Images, FileIO
# Define the size and PSF parameters
sz = (128, 128, 128)
pp_illu = PSFParams(0.5, 0.5, 1.52)

# Generate coherent PSF, compute its intensity and obtain the OTF
h2d = sum(abs2.(sum(apsf(sz, pp_illu; sampling=(0.20, 0.2, 0.200)), dims=1)), dims=4)[1, :, :, 1]
otf2d = fft(h2d)

# Perform SVD on the OTF
F = svd(otf2d)

# Reconstruct full OTF from SVD components
# We use the feature of seperability to the full matrix otf2d to the sigma_i*U and V.
max_components = 4
otf_comp_x = [F.U[:, n] for n = 1:max_components]
otf_comp_y = [(F.S[n] * F.Vt[n, :])' for n = 1:max_components]
A_red = sum(otf_comp_x[n] .* otf_comp_y[n] for n = 1:max_components)

# Compute PSF components from the OTF components
psf_comp_x = [real.(ifft(Array(otf_comp_x[n]), 1)) for n = 1:max_components]
psf_comp_y = [real.(ifft(Array(otf_comp_y[n]), 2)) for n = 1:max_components]

# Simulate a detection PSF
pp_det = PSFParams(0.5, 0.8, 1.52)
h_det = psf(sz, pp_det; sampling=(0.20, 0.2, 0.200))

# Generate a 3D object
obj = filaments3D(sz)

# Function to compute the lightsheet image by performing a 2d convolution and a 1d integral
function compute_lightsheet_image(obj, psf_comp_x, psf_comp_y, h_det, max_components, sz, z_0)
    # Initialize the resulting image
    f = zeros(Float64, sz[1], sz[2], sz[3])

    # Loop over each component
    for i in 1:max_components
        # Reorient the illumination component in the x-direction
        illu_x = reorient(psf_comp_y[i], Val(1))

        # Initialize the integral result for this component
        integral_result = zeros(Float64, sz[1], sz[2], sz[3])

        # Loop over the z-direction
        for k in 1:sz[3]
            w = k - z_0 + div(sz[3], 2)  # Adjust the z index to be centered around z_0

            # Ensure w is within bounds
            if w > 0 && w <= sz[3]
                # Reorient the illumination component in the z-direction
                illu_z = reorient(psf_comp_x[i], Val(3))[:, :, w]

                # Compute K as the product of the illumination components and the object
                K = illu_z .* obj[:, :, k]

                # Convolve K with the detection PSF
                if w <= sz[3]
                    K_conv = conv_psf(K, h_det[:, :, w])
                else
                    K_conv = conv_psf(K, h_det[:, :, sz[3]])
                end

                # Accumulate the convolution result
                integral_result[:, :, k] += K_conv
            end
        end

        # Accumulate the integral result into the final image
        f += illu_x .* integral_result
    end

    return f
end

# Compute the lightsheet image, set z_0=64
result_image = compute_lightsheet_image(obj, psf_comp_x, psf_comp_y, h_det, max_components, sz, 64)
@vv result_image


# Reconstruct a blurred model using the forward model, by iterating z_0
function reconstruct_3d_model(obj, psf_comp_x, psf_comp_y, h_det, max_components, sz)
    Nz = sz[3]
    # Initialize a 3D array to store the reconstructed image
    result_image_3d = zeros(Float64, sz[1], sz[2], Nz)

    # Iterate over z_0 from 1 to Nz
    for z_0 in 1:Nz
        result_image_3d[:, :, z_0] = sum(compute_lightsheet_image(obj, psf_comp_x, psf_comp_y, h_det, max_components, sz, z_0), dims=3)
    end

    return result_image_3d
end

# Compute the 3D lightsheet image
result_image_3d = reconstruct_3d_model(obj, psf_comp_x, psf_comp_y, h_det, max_components, sz)

# Print the result image size
println("Result image size: ", size(result_image_3d))

@vv result_image_3d
