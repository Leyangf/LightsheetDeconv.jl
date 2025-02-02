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

    function perform_deconvolution(nimg, psf_comp_x, psf_comp_z, h_det, bwd_components;
                                   iterations=50, goods_weight=0.01)
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
            reg_loss = goods_weight * goods_roughness(current_obj[])
            return data_loss + reg_loss
        end

        start_val = (obj = Positive(mean(nimg) .* ones(Float64, size(nimg))),)

        res, myloss = optimize_model(start_val, forward_model, nimg, composite_loss;
                                     iterations=iterations)
        return res
    end    

end # module
