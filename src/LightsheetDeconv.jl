module LightSheetDeconv
    export perform_deconvolution

    include("LightsheetFwd.jl")
    using .LightSheetSimulation
    using InverseModeling
    using PointSpreadFunctions, FourierTools, Statistics, LinearAlgebra

    function goods_roughness(img::AbstractArray)
        total = zero(eltype(img))
        for d in 1:ndims(img)
            r = diff(img, dims=d)
            total += sum(r .^ 2)
        end
        return total
    end

    function tv_roughness(img::AbstractArray)
        total = zero(eltype(img))
        for d in 1:ndims(img)
            r = diff(img, dims=d)
            total += sum(abs.(r))
        end
        return total
    end

    function tikhonov_regularization(img::AbstractArray)
        return sum(abs2, img)
    end

    function perform_deconvolution(nimg, psf_comp_x, psf_comp_z, h_det, bwd_components;
        iterations=50, reg_weight=0.01, reg_type="goods")
        
        sz = size(nimg)
        current_obj = Ref{Any}(nothing)

        function forward_model(g)
            obj = g(:obj)
            current_obj[] = obj
            return LightSheetSimulation.simulate_lightsheet_image(obj, sz,
                                           psf_comp_x, psf_comp_z,
                                           h_det, bwd_components)
        end

        function composite_loss(simulated, measured, extra)
            data_loss = loss_poisson_pos(simulated, measured, extra)
            reg_loss = if reg_type == "goods"
                reg_weight * goods_roughness(current_obj[])
            elseif reg_type == "tv"
                reg_weight * tv_roughness(current_obj[])
            elseif reg_type == "tikhonov"
                reg_weight * tikhonov_regularization(current_obj[])
            else
                error("Unknown regularizer type: $reg_type.")
            end
            return data_loss + reg_loss
        end

        start_val = (obj = Positive(mean(nimg) .* ones(Float32, size(nimg))),)

        res, myloss = optimize_model(start_val, forward_model, nimg, composite_loss;
                  iterations=iterations)
        
        return res
    end    
end # module
