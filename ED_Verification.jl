using Distributed
using ITensors, ITensorMPS
using LinearAlgebra
using Plots, ColorSchemes, LaTeXStrings
default(color_palette = :okabe_ito)
using Printf
using Base64
using Dates
using JLD2 
include("EDModule.jl")
using .EDModule 

# ---------------------------------
# Cluster Management 
# ---------------------------------
N_SIM = 5

n_main = 2
n_compute1 = 4
n_compute2 = 2 
n_compute3 = 4

n_remote_workers = n_compute1 + n_compute2 + n_compute3

@info "Starting Worker Processes"
rmprocs(workers()) 

@info addprocs(n_main) ## Use only this if running locally
@info addprocs([("alex@alex-sim1", n_compute1)]; shell=:wincmd, tunnel=false, exename="C:\\Users\\Alex\\AppData\\Local\\Programs\\Julia-1.11.6\\bin\\julia.exe")
@info addprocs([("alex@alex-compute2", n_compute2)]; shell=:wincmd, tunnel=false, exename="C:\\Users\\Alex\\.julia\\juliaup\\julia-1.11.6+0.x64.w64.mingw32\\bin\\julia.exe")
@info addprocs([("alex@alex-SIM3", n_compute3)]; shell=:wincmd, tunnel=false, exename="C:\\Users\\Alex\\AppData\\Local\\Programs\\Julia-1.11.6\\bin\\julia.exe", env=Dict("JULIA_CLUSTER_COOKIE" => "nxN63uJZx40j4dR3"))

@everywhere begin
  using LinearAlgebra, SparseArrays, Random
  using ITensors, ITensorMPS 
  using Dates, Printf
  using KrylovKit, LinearMaps 
  include("EDModule.jl")
  using .EDModule
end

try
  BLAS.set_num_threads(t_main)
  catch
   @warn "Could not set BLAS threads"
end

# ---------------------------------
# System & Scan Parameters
# ---------------------------------
N = 3 # Numer of Fermion Flavors
M = 6 # Number of Sites 

# Scan Parameters
TRS = 0 # Imaginary Component of Coupling: J = J0 * (1 + TRS * im)
mu = 1.0 # Chemical Potential 
W = 0 # Disorder (Not Currently Implemented)
J = 10 # Coupling Strength

# Choose Dense / Krylov Parameters
krylov_k = 1
dense_cutoff = 6300 
krylov_tol = 1e-10
krylov_maxiter = 900_000

# Calculate the unnormalized integer target Q for dense calculation
Q_target_int = BigInt(3)^(M - 1)
# Q_target_int = Int(floor(N*(BigInt(3)^(M)-1)/8))
@info "Target Q = $Q_target_int"
# ---------------------------------
# Run Simulations
# ---------------------------------
start_time = now()
@info "Starting distributed ED simulations at $(Dates.format(start_time, "U d, I:M p"))"

ED_results = EDModule.run_ed_simulation(
  M, N, J, mu, 
  W;
  krylov_k=krylov_k,
  dense_cutoff=dense_cutoff,
  krylov_tol=krylov_tol,
  krylov_maxiter=krylov_maxiter,
  target_Q_integer=Q_target_int
)

GC.gc()

@info "Master: Returning results for N=$(N), M=$(M), and J=$J"

ED_point = (
  GS_energy=ED_results["GS_energy"],
  GS_charge=ED_results["GS_charge"],
  GS_site_densities=ED_results["GS_site_densities"],
  GS_mode_densities=ED_results["GS_mode_densities"],
  all_energies=ED_results["all_energies"],
  all_charges=ED_results["all_charges"]
)

end_time = now()
elapsed_time = Dates.canonicalize(Dates.CompoundPeriod(end_time - start_time))
@info "\nAll distributed ED calculations complete. Total elapsed time: $(elapsed_time)"

# Save Data
data_output_dir = "Plots/Verification/N_SIM$(N_SIM)/data"
mkpath(data_output_dir)
output_filename = joinpath(data_output_dir, "full_data_N$(N)_M$(M)_NSIM$(N_SIM).jld2")
@save output_filename ED_point
@info "Data saved to: $(output_filename)"

# ---------------------------------
# Data Analysis & Plotting
# ---------------------------------
@info "Data Plotting Section"

# Plotting Data
plot_output_dir = "Plots/Verification/N_SIM$(N_SIM)/"
mkpath(plot_output_dir)

println("Ground state energy E = $(ED_point.GS_energy)")

# 1. Energy Spectrum Plotting (Target Q Sector)
# Normalized target Q: Q_target = (3^(M - 1)) / (N * sum(3^(M - m) for m in 1:M))
Q_target = (Q_target_int) / (N * sum(3^(M - m) for m in 1:M))
ii_Q_target = findall(q -> isapprox(q, Q_target, atol=1e-5), ED_point.all_charges)

if !isempty(ii_Q_target)
    sector_energies = ED_point.all_energies[ii_Q_target]
    sorted_energies = sort(sector_energies) 

    energy_plot = scatter(
        1:length(sorted_energies),
        sorted_energies,
        label = "Energy Eigenvalues",
        title = "Energy Spectrum (N=$(N), M=$(M), J=$(J)) \n for Q = $(round(Q_target, digits=3))",
        xlabel = "Eigenvalue Index",
        ylabel = "Energy",
       
        marker = :circle,
        markersize = 4,
        markerstrokewidth = 2,
        markerstrokealpha = 1,
        linecolor = nothing,
        legend = false
    )

    savefig(energy_plot, "$plot_output_dir/Energies_N$(N)_M$(M)_J$(J).pdf")
    println("Saved Energies Plot for the target Q sector.")
else
    println("No energy data found for the target Q sector to plot.")
    sorted_energies = ED_point.all_energies 
end

# Implement: Entanglement Calc

# 2. Final 
output_filename_spectrum = joinpath(data_output_dir, "JL_ED_Data_N$(N)_M$(M)_J$(J).jld2")
@save output_filename_spectrum sorted_energies
@info "Sorted spectrum data saved to: $(output_filename_spectrum)"


# README Generation
readme_message = """
Verifying against Bo-Ting's code. In Q=1/4 CS
"""

global params_str = """
Simulation Parameters:
----------------------
Number of Simulations (N_SIM): $(N_SIM)
Total Workers: $(n_remote_workers+n_main)

N: $(N)
M: $(M)
TRS (Time Reversal Symmetry parameter): $(TRS)
mu (Chemical Potential): $(mu)

Krylov State Cutoff = $krylov_k
Dense-Krylov Cutoff Dimension = $dense_cutoff
Krylov Tolerance = $krylov_tol
Krylov Max Iterations = $krylov_maxiter
"""

global params_str
params_str *= """
Simulation Start Time: $(start_time)
Simulation End Time: $(end_time)
Total Elapsed Time: $(elapsed_time)
"""

readme_path = joinpath(plot_output_dir, "readme.txt")
open(readme_path, "w") do io
  write(io, "Simulation Message:\n")
  write(io, readme_message)
  write(io, "\n")
  write(io, params_str)
end

@info "Generated README file: $readme_path"
rmprocs(workers())
@info "Worker processes removed."
@info "All tasks completed successfully!"