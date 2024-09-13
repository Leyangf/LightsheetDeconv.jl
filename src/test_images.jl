module TestImages
    export create_full_bead_image, crop_subregion, create_evenly_beads, compute_NCC
    using Statistics


    # Function to create a 3D image with a single bead at the specified position and intensity
    function create_full_bead_image(sz, bead_position::Tuple{Int,Int,Int}, bead_intensity::Float64)
        img = zeros(Float64, sz)
        x, y, z = bead_position
        img[x, y, z] = bead_intensity
        return img
    end

    # Function to crop a subregion from a 3D image centered at 'center_pos' with size 'sub_sz'
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

    # Function to generate evenly spaced beads in a 3D space with specified intensity
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

    # Function to compute the normalized cross-correlation (NCC) between two 3D arrays
    function compute_NCC(array1::Array{Float32, 3}, array2::Array{Float32, 3})
        # Ensure the arrays have the same dimensions
        if size(array1) != size(array2)
            throw(ArgumentError("Arrays must have the same dimensions"))
        end

        # Compute the means of the arrays
        mean1 = mean(array1)
        mean2 = mean(array2)

        # Subtract the means from the arrays to create zero-mean arrays
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

end