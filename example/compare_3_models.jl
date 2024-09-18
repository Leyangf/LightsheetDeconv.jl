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
sheet_model_gaussian = simulate_elliptic_gaussian(sz, 0.488, 0.33, 1.52; sampling=(0.2, 0.2, 0.2))
sheet_model_gaussian = normalize_intensity(sheet_model_gaussian)

# Display the Gaussian-based light sheet model
volume(sheet_model_gaussian)

# ----------------------------------------------------------------------
# Calculate Full Width at Half Maximum (FWHM) for each model
# ----------------------------------------------------------------------

# Calculate FWHM for diffraction, DSLM, and Gaussian models along x-axis
fwhm_diffraction = calculate_fwhm_for_model(sheet_model_diffraction, sz, :x)
fwhm_gaussian = calculate_fwhm_for_model(sheet_model_gaussian, sz, :x)
fwhm_dslm = calculate_fwhm_for_model(sheet_model_dslm, sz, :x)

# ----------------------------------------------------------------------
# Plot Thickness as a Function of Propagation Distance
# ----------------------------------------------------------------------

# Define x-axis (propagation distance) in microns based on sampling (0.2 µm per voxel)
x_axis = range(-sz[1] * 0.2 / 2, sz[1] * 0.2 / 2, length=sz[1])

# Plot thickness (FWHM) for diffraction, DSLM, and Gaussian models
Plots.plot(x_axis, fwhm_diffraction, label="Diffraction Model", xlabel="Propagation Distance (µm)", ylabel="Thickness (FWHM in z) (µm)", legend=:topright)
Plots.plot!(x_axis, fwhm_gaussian, label="Gaussian Model")
Plots.plot!(x_axis, fwhm_dslm, label="DSLM Model")