#= Implement a gibbs sampler to sample from a multivariate normal distribution.
Breaks up a n length vector into p vectors of equal length m. Uses known
analytical conditional distributions to estimate multivariate normal
distribution. =#

#load packages
using Distributions, Statistics, LinearAlgebra, Random

#Set random seed
Random.seed!(8675309)

# -----------------------------------------------------------------------------
# Function
# -----------------------------------------------------------------------------

function condcov( S::Matrix{<:AbstractFloat}, 
                  jinds::UnitRange{Int}, 
                  notjinds::Vector{Int} )

    # pieces for conditional covariance matrix
    Σ11 = S[jinds, jinds]
    Σ12 = S[jinds, notjinds]
    Σ21 = Σ12'
    Σ22 = S[notjinds, notjinds]

    Σ12_invΣ22 = Σ12 / Σ22 

    #Covariance for sampling distribution
    Σ = Σ11 - Σ12_invΣ22*Σ21
    Σ = 0.5*(Σ + Σ')

    return (Σ, Σ12_invΣ22)

end

# single draw
function MvNormal_Gibbs(mu::Vector{<:AbstractFloat},  # mean
                        S::Matrix{<:AbstractFloat},   # covariance
                        x::Vector{<:AbstractFloat},   # current values
                        nsub::Int)                    # number of subvectors
                                            

    #Check nsub
    nfull   = length(mu)
    sub_len = nfull/nsub::Int

    for j in 1:nsub # index of subvector

        #indices of j-th subvector
        start = Int((j - 1)*sub_len + 1)
        stop  = Int(j*sub_len)

        # more helper indices
        jinds    = start:stop
        notjinds = [1:(start - 1); (stop + 1):nfull]

        # Σ
        (Σ, Σ12_invΣ22) = condcov(S, jinds, notjinds)

        # μ
        μ = mu[jinds] + Σ12_invΣ22*( x[notjinds] - mu[notjinds] )

        # Sample
        x[jinds] = rand(MvNormal(μ, Σ))

    end

    return x
end

# multiple iterations
function MvNormal_Gibbs( nsim::Int, 
                         mean_vector::Vector{<:AbstractFloat},
                         covariance::Matrix{<:AbstractFloat}, 
                         nsub::Int )

    #preallocate
    samp      = ones( length(mean_vector), nsim )
    samp[:,1] = mean_vector

    #Gibbs sample multiple draws
    for i in 2:nsim
        
        samp[:,i] = MvNormal_Gibbs(mean_vector, covariance, samp[:,i-1], nsub)
        
    end

    return samp

end

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------

nsim        = 100000::Int
burn        = 0::Int

vec_len     = 20::Int
nsub        = 5::Int

# random mean and covariance
mean_vector = rand(vec_len)
prep        = rand(vec_len,vec_len)
covariance  = (prep + prep')/2 + vec_len*Matrix{Float64}(I, vec_len, vec_len )

# -----------------------------------------------------------------------------
# Example
# -----------------------------------------------------------------------------

samp = MvNormal_Gibbs(nsim, mean_vector, covariance, nsub)

#Check difference in mean
mean_check = mean(samp[:,burn+1:end] , dims = 2) - mean_vector
cov_check  = cov(samp[:,burn+1:end], dims=2) - covariance
