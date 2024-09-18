module TestFunctions
    export create_full_bead_image, crop_subregion, create_evenly_beads, compute_NCC, normalize_intensity, create_sheet_model, calculate_fwhm_for_model, simulate_elliptic_gaussian
    using Statistics

    # ----------------------------------------------------------------------
    # Function: create_full_bead_image
    # Description: Creates a 3D image with a single bead at the specified position and intensity.
    # ----------------------------------------------------------------------
    function create_full_bead_image(sz, bead_position::Tuple{Int, Int, Int}, bead_intensity::Float64)
        img = zeros(Float64, sz)
        x, y, z = bead_position
        img[x, y, z] = bead_intensity
        return img
    end

    # ----------------------------------------------------------------------
    # Function: crop_subregion
    # Description: Crops a subregion from a 3D image centered at 'center_pos' with size 'sub_sz'.
    # ----------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
    # Function: create_evenly_beads
    # Description: Generates evenly spaced beads in a 3D space with specified intensity.
    # ----------------------------------------------------------------------
    function create_evenly_beads(sz, num_beads::Int, bead_intensity::Float64)
        img = zeros(Float64, sz)
        
        # Calculate the step size for each dimension based on cube root of the number of beads
        num_per_dim = ceil(Int, num_beads^(1/3))
        x_step = max(floor(Int, sz[1] / num_per_dim), 1)
        y_step = max(floor(Int, sz[2] / num_per_dim), 1)
        z_step = max(floor(Int, sz[3] / num_per_dim), 1)

        # Cartesian index generator with steps
        cartesian_indices = CartesianIndices((20:x_step:sz[1] - 20, 20:y_step:sz[2] - 20, 20:z_step:sz[3] - 20))

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

    # ----------------------------------------------------------------------
    # Function: compute_NCC
    # Description: Computes the normalized cross-correlation (NCC) between two 3D arrays.
    # ----------------------------------------------------------------------
    function compute_NCC(array1::Array{Float32, 3}, array2::Array{Float32, 3})
        if size(array1) != size(array2)
            throw(ArgumentError("Arrays must have the same dimensions"))
        end

        # Compute means of the arrays
        mean1 = mean(array1)
        mean2 = mean(array2)

        # Subtract means to create zero-mean arrays
        array1_zero_mean = array1 .- mean1
        array2_zero_mean = array2 .- mean2

        # Compute numerator (element-wise product of zero-mean arrays)
        numerator = sum(array1_zero_mean .* array2_zero_mean)

        # Compute denominator (variance terms)
        sum_sq_array1 = sum(array1_zero_mean .^ 2)
        sum_sq_array2 = sum(array2_zero_mean .^ 2)
        denominator = sqrt(sum_sq_array1 * sum_sq_array2)

        if denominator == 0
            throw(ArgumentError("Denominator is zero, which means one or both arrays have no variance."))
        end

        # Return the normalized cross-correlation
        return numerator / denominator
    end

    # ----------------------------------------------------------------------
    # Function: normalize_intensity
    # Description: Normalizes the intensity of the light sheet by dividing each value by the total intensity.
    # ----------------------------------------------------------------------
    function normalize_intensity(light_sheet::AbstractArray)
        total_intensity = sum(light_sheet)
        light_sheet .= light_sheet ./ total_intensity  # In-place normalization
        return light_sheet
    end

    # ----------------------------------------------------------------------
    # Function: create_sheet_model
    # Description: Creates a 3D light sheet model from a 2D profile.
    # ----------------------------------------------------------------------
    function create_sheet_model(profile::AbstractArray, sz::NTuple{3, Int})
        sheet_model_array = reshape(profile, sz[1], 1, sz[2])
        sheet_model = repeat(sheet_model_array, 1, sz[3], 1)  # Repeat along y-axis
        return sheet_model
    end

    # ----------------------------------------------------------------------
    # Function: calculate_fwhm_for_model
    # Description: Calculates the Full Width at Half Maximum (FWHM) for a light sheet model along a given axis.
    # ----------------------------------------------------------------------
    function calculate_fwhm_for_model(model::AbstractArray, sz::NTuple{3, Int}, axis::Symbol=:x)
        fwhm_values = []

        # Calculate FWHM for a 1D intensity profile
        function calculate_fwhm(profile::AbstractVector{<:AbstractFloat})
            max_val = maximum(profile)
            half_max = max_val / 2
            idx_above_half = findall(x -> x >= half_max, profile)

            return (length(idx_above_half) < 2) ? 0.0 : abs(idx_above_half[end] - idx_above_half[1])
        end

        # Slice model and calculate FWHM depending on axis
        if axis == :x
            for i in 1:sz[1]
                profile = model[i, div(sz[2], 2), :]
                push!(fwhm_values, calculate_fwhm(profile))
            end
        elseif axis == :y
            for j in 1:sz[2]
                profile = model[div(sz[1], 2), j, :]
                push!(fwhm_values, calculate_fwhm(profile))
            end
        elseif axis == :z
            for k in 1:sz[3]
                profile = model[div(sz[1], 2), :, k]
                push!(fwhm_values, calculate_fwhm(profile))
            end
        else
            error("Invalid axis. Please use :x, :y, or :z.")
        end

        return fwhm_values
    end

    # ----------------------------------------------------------------------
    # Function: simulate_elliptic_gaussian
    # Description: Simulates an elliptic Gaussian beam for light sheet microscopy.
    # ----------------------------------------------------------------------
    function simulate_elliptic_gaussian(sz, λ, NA, n; sampling=(0.2, 0.2, 0.2), width_factor)
        W_z0 = width_factor * λ / (π * NA * n)  # Beam waist in z direction
        W_y0 = W_z0 * 200        # Beam waist in y direction (scaled for anisotropy)

        # Rayleigh ranges
        z_z0 = π * W_z0^2 * n / λ
        z_y0 = π * W_y0^2 * n / λ

        # Create coordinate grids (x, y, z)
        x_range = range(-sz[1] * sampling[1] / 2, sz[1] * sampling[1] / 2, length=sz[1])
        y_range = range(-sz[2] * sampling[2] / 2, sz[2] * sampling[2] / 2, length=sz[2])
        z_range = range(-sz[3] * sampling[3] / 2, sz[3] * sampling[3] / 2, length=sz[3])

        # Beam widths as functions of propagation distance
        W_z = x -> W_z0 * sqrt(1 + (x / z_z0)^2)
        W_y = x -> W_y0 * sqrt(1 + (x / z_y0)^2)

        # Initialize PSF array
        psf = zeros(Float64, sz...)

        # Iterate over x, y, z positions to compute the intensity
        for (i, x) in enumerate(x_range)
            for (j, y) in enumerate(y_range)
                for (k, z) in enumerate(z_range)
                    W_zx = W_z(x)
                    W_yx = W_y(x)
                    psf[i, j, k] = (W_z0 / W_zx) * (W_y0 / W_yx) * exp(-2 * z^2 / W_zx^2) * exp(-2 * y^2 / W_yx^2)
                end
            end
        end

        return psf
    end

end