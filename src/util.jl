module TestFunctions
    export create_evenly_beads, compute_NCC
    using Statistics

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
        cartesian_indices = CartesianIndices((10:x_step:sz[1] - 10, 10:y_step:sz[2] - 10, 10:z_step:sz[3] - 10))

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
    # Description: Computes the normalized cross-correlation (NCC) between two arrays.
    # ----------------------------------------------------------------------
    function compute_NCC(array1::AbstractArray{Float32, N}, array2::AbstractArray{Float32, N}) where N
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

end