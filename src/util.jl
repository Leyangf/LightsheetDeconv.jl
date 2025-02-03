module TestFunctions
    export create_evenly_beads, compute_NCC, ssim3d
    using Statistics
    using ImageFiltering

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


    function ssim3d(img1::Array{Float32,3}, img2::Array{Float32,3}; window_size=(11,11,11), sigma=1.5, L=1.0)
        @assert size(img1) == size(img2) "Input volumes must have the same dimensions."

        # Constants for SSIM
        c1 = (0.01 * L)^2
        c2 = (0.03 * L)^2

        # Create a 3D Gaussian kernel.
        # The kernel is separable; KernelFactors.gaussian returns a tuple of 1D kernels.
        kernel = KernelFactors.gaussian((sigma, sigma, sigma), window_size)

        # Compute local means via convolution with boundary handling.
        μ1 = imfilter(img1, kernel, Pad(:replicate))
        μ2 = imfilter(img2, kernel, Pad(:replicate))

        # Precompute squares and products.
        μ1_sq = μ1 .^ 2
        μ2_sq = μ2 .^ 2
        μ1μ2 = μ1 .* μ2

        # Compute local variances and covariance.
        σ1_sq = imfilter(img1 .^ 2, kernel, Pad(:replicate)) .- μ1_sq
        σ2_sq = imfilter(img2 .^ 2, kernel, Pad(:replicate)) .- μ2_sq
        σ12   = imfilter(img1 .* img2, kernel, Pad(:replicate)) .- μ1μ2

        # Compute the SSIM map over the volume.
        numerator   = (2 .* μ1μ2 .+ c1) .* (2 .* σ12 .+ c2)
        denominator = (μ1_sq .+ μ2_sq .+ c1) .* (σ1_sq .+ σ2_sq .+ c2)
        ssim_map    = numerator ./ denominator

        # Return the mean SSIM over all voxels.
        return mean(ssim_map)
    end

end