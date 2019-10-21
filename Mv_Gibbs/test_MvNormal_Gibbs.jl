using Distributions, Statistics, LinearAlgebra, Random

#Set random seed
Random.seed!(8675309)

# -----------------------------------------------------------------------------
# File I/O settings and load functions
# -----------------------------------------------------------------------------

if homedir() == "C:\\Users\\tgwin"

  location_data = "C:\\Users\\tgwin\\OneDrive\\Fed\\Mark Bognanni"

elseif homedir() == "C:\\Users\\Mark"

  location_data = "C:\\Users\\Mark\\Documents\\git_repos\\Sharing_Mark"

end

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

cd(location_data)

# Covariance functions
include("helpers_indexing.jl")

# Covariance functions
include("Multivariate_Normal_Gibbs_Covariance.jl")

# Precision functions
include("Multivariate_Normal_Gibbs_Precision.jl")

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------

nsim        = 500000::Int
burn        = 0::Int

vec_len     = 20::Int
nsub        = 5::Int

# random mean and covariance
mean_vector = rand(vec_len)
prep        = rand(vec_len,vec_len)
covariance  = (prep + prep')/2 + vec_len*Matrix{Float64}(I, vec_len, vec_len )
prec        = inv(covariance)

# -----------------------------------------------------------------------------
# Run functions
# -----------------------------------------------------------------------------

samp_cov  = MvNormal_Gibbs_Covariance(nsim, mean_vector, covariance, nsub)
samp_prec = MvNormal_Gibbs_Precision(nsim, mean_vector, prec, nsub)

# -----------------------------------------------------------------------------
# Test results
# -----------------------------------------------------------------------------

mean_diff = mean(samp_cov, dims = 2) - mean(samp_prec, dims = 2)

