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

    function tgv_roughness(img::AbstractArray; alpha0=0.5, alpha1=0.5)
        total1 = zero(eltype(img))
        total2 = zero(eltype(img))
        for d in 1:ndims(img)
            r1 = diff(img, dims=d)
            total1 += sum(abs.(r1))
            r2 = diff(r1, dims=d)
            total2 += sum(abs.(r2))
        end
        return alpha1 * total1 + alpha0 * total2
    end

    function huber_tv_roughness(img::AbstractArray; delta=0.1)
        total = zero(eltype(img))
        for d in 1:ndims(img)
            r = diff(img, dims=d)
            total += sum(ifelse.(abs.(r) .< delta, (r .^ 2) / (2 * delta), abs.(r) .- delta / 2))
        end
        return total
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
elseif reg_type == "tgv"
reg_weight * tgv_roughness(current_obj[])
elseif reg_type == "huber_tv"
    reg_weight * huber_tv_roughness(current_obj[])
else
error("Unknown regularizer type: $reg_type. Use \"goods\", \"tv\", or \"tgv\", or \"huber_tv\".")
end
return data_loss + reg_loss
end

start_val = (obj = Positive(mean(nimg) .* ones(Float32, size(nimg))),)

res, myloss = optimize_model(start_val, forward_model, nimg, composite_loss;
          iterations=iterations)
return res
end    
end # module
