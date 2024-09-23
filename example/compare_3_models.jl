# ----------------------------------------------------------------------
# Import necessary libraries and set up configurations
# ----------------------------------------------------------------------
include("../src/util.jl")
using .TestFunctions
using PointSpreadFunctions
using Plots, GLMakie

# Define size and Point Spread Function (PSF) parameters
sz = (128, 128, 128)
pp_illu = PSFParams(0.488, 0.33, 1.52)  # Illumination parameters for PSF

# ----------------------------------------------------------------------
# Diffraction-based Light Sheet Simulation
# ----------------------------------------------------------------------

# Generate 3D coherent PSF and sum along x-dimension to get diffraction profile
h2d = sum(abs2.(sum(apsf(sz, pp_illu; sampling=(0.20, 0.2, 0.2)), dims=1)), dims=4)[1, :, :, 1]
profile_diffraction = transpose(h2d)

# Create and normalize the sheet model based on diffraction profile
sheet_model_diffraction = create_sheet_model(profile_diffraction, sz)
sheet_model_diffraction = normalize_intensity(sheet_model_diffraction)

# Display the diffraction-based light sheet model
volume(sheet_model_diffraction)

# ----------------------------------------------------------------------
# DSLM-based Light Sheet Simulation
# ----------------------------------------------------------------------

# Generate 3D incoherent PSF and sum over x-dimension to simulate DSLM profile
h3d = PointSpreadFunctions.psf(sz, pp_illu; sampling=(0.2, 0.2, 0.2))
h2d_dslm = sum(h3d, dims=1)[1, :, :]
profile_dslm = transpose(h2d_dslm)

# Create and normalize the sheet model based on DSLM profile
sheet_model_dslm = create_sheet_model(profile_dslm, sz)
sheet_model_dslm = normalize_intensity(sheet_model_dslm)

# Display the DSLM-based light sheet model
volume(sheet_model_dslm)

# ----------------------------------------------------------------------
# Gaussian Approximation of Light Sheet
# ----------------------------------------------------------------------

# Simulate Gaussian-based light sheet
sheet_model_gaussian = simulate_elliptic_gaussian(sz, 0.488, 0.33, 1.52; sampling=(0.2, 0.2, 0.2), width_factor=1.8) # Introduce width factor, but still thicker when same FOV
sheet_model_gaussian = normalize_intensity(sheet_model_gaussian)

# Display the Gaussian-based light sheet model
volume(sheet_model_gaussian)

# ----------------------------------------------------------------------
# Calculate Full Width at Half Maximum (FWHM) for each model, and plot Thickness as a Function of Propagation Distance
# ----------------------------------------------------------------------

# Calculate FWHM for diffraction, DSLM, and Gaussian models along x-axis
fwhm_diffraction = calculate_fwhm_for_model(sheet_model_diffraction, sz, :x)
fwhm_gaussian = calculate_fwhm_for_model(sheet_model_gaussian, sz, :x)
fwhm_dslm = calculate_fwhm_for_model(sheet_model_dslm, sz, :x)

# Define x-axis (propagation distance) in microns based on sampling (0.2 µm per voxel)
x_axis = range(-sz[1] * 0.2 / 2, sz[1] * 0.2 / 2, length=sz[1])

# Plot thickness (FWHM) for diffraction, DSLM, and Gaussian models
Plots.plot(x_axis, fwhm_diffraction, label="Diffraction Model", xlabel="Propagation Distance (µm)", ylabel="Thickness (FWHM in z) (µm)", legend=:topright)
Plots.plot!(x_axis, fwhm_gaussian, label="Gaussian Model")
Plots.plot!(x_axis, fwhm_dslm, label="DSLM Model")

# Plot heatmap for diffraction, DSLM, and Gaussian models
Plots.plot(
    Plots.heatmap(normalize_intensity(h2d),title="Diffraction Model with Cylindrical Lens"), 
    Plots.heatmap(normalize_intensity(h2d_dslm),title="Diffraction Model with DSLM"), 
    Plots.heatmap(normalize_intensity(transpose(sheet_model_gaussian[:, sz[2] ÷ 2, :])), 
            title="Elliptical Gaussian Model"), 
    layout = (3, 1),
    size = (500, 700) 
)

# Plot intensity profiles for diffraction, DSLM, and Gaussian models
profile_h2d = h2d[:, size(h2d, 2) ÷ 2]  # Middle column of h2d
profile_h2d_dslm = h2d_dslm[:, size(h2d_dslm, 2) ÷ 2]  # Middle column of h2d_dslm
profile_gaussian = sheet_model_gaussian[:, size(sheet_model_gaussian, 2) ÷ 2, size(sheet_model_gaussian, 3) ÷ 2]  # Middle slice of Gaussian model
Plots.plot(
    profile_h2d, label="Cylindrical Lens", lw=2, color=:blue,
    xlabel="Position", ylabel="Intensity", title="Intensity Profiles of Light Sheets"
)
Plots.plot!(profile_h2d_dslm, label="DSLM", lw=2, color=:green)
Plots.plot!(profile_gaussian, label="Elliptical Gaussian", lw=2, color=:red)