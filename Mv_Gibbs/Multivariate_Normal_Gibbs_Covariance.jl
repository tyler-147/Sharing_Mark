#= Implement a gibbs sampler to sample from a multivariate normal distribution.
Breaks up a n length vector into p vectors of equal length m. Uses known
analytical conditional distributions to estimate multivariate normal
distribution, taking covariance matrix as an input. Packages required:
Distributions, Statistics, LinearAlgebra,and Random.=#

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

function condmean( x::Vector{<:AbstractFloat},
                   mu::Vector{<:AbstractFloat},
                   Σ12_invΣ22::Matrix{<:AbstractFloat},
                   jinds::UnitRange{Int},
                   notjinds::Vector{Int} )

    μ = mu[jinds] + Σ12_invΣ22*( x[notjinds] - mu[notjinds] )

    return μ
end

function cond_mean_cov( x::Vector{<:AbstractFloat},
                        mu::Vector{<:AbstractFloat},
                        S::Matrix{<:AbstractFloat},
                        jinds::UnitRange{Int},
                        notjinds::Vector{Int} )

    # Σ[j|notj]
    (Σ, Σ12_invΣ22) = condcov(S, jinds, notjinds)

    # μ[j|notj]
    μ = condmean(x, mu, Σ12_invΣ22, jinds, notjinds)

    return (μ, Σ)
end

# single draw
function MvNormal_Gibbs_Covariance(mu::Vector{<:AbstractFloat},  # mean
                        S::Matrix{<:AbstractFloat},   # covariance
                        x::Vector{<:AbstractFloat},   # current values
                        nsub::Int)                    # number of subvectors

    #Check nsub
    nfull   = length(mu)::Int
    sub_len = div(nfull, nsub)::Int

    for j in 1:nsub # index of subvector

        # set j-th block indices
        (jinds, notjinds) = setjinds(j, sub_len, nfull)

        # compute conditional mean and covariance
        (μ, Σ) = cond_mean_cov(x, mu, S, jinds, notjinds)

        # Sample x[j] | x[notj]
        x[jinds] = rand(MvNormal(μ, Σ))

    end

    return x
end

# multiple iterations
function MvNormal_Gibbs_Covariance( nsim::Int,
                         mean_vector::Vector{<:AbstractFloat},
                         covariance::Matrix{<:AbstractFloat},
                         nsub::Int )

    #preallocate
    samp      = ones( length(mean_vector), nsim )
    samp[:,1] = mean_vector

    #Gibbs sample multiple draws
    for i in 2:nsim

        samp[:,i] = MvNormal_Gibbs_Covariance(mean_vector, covariance, samp[:,i-1], nsub)

    end

    return samp

end
