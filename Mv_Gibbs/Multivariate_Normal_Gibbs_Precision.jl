#= Implement a gibbs sampler to sample from a multivariate normal distribution.
Breaks up a n length vector into p vectors of equal length m. Uses known
analytical conditional distributions to estimate multivariate normal
distribution, taking precision matrix as an input. Packages required:
Distributions, Statistics, LinearAlgebra,and Random.=#

function setjinds( j::Int, sub_len::Int, full_len::Int )

    #indices of j-th subvector
    start = Int((j - 1)*sub_len + 1)
    stop  = Int(j*sub_len)

    jinds    = start:stop                   # UnitRange
    notjinds = setdiff(1:full_len, jinds)   # Vector of Ints

    return (jinds, notjinds)
end

function condcov_prec(S::Matrix{<:AbstractFloat},
                      jinds::UnitRange{Int},
                      notjinds::Vector{Int})

    #pieces of the precision matrix
    Λ11 = S[jinds, jinds]
    Λ12 = S[jinds, notjinds]

    #estimate conditional sigma
    Σ = inv(Λ11)
    Σ = .5*( Σ + Σ' )
    
    return (Σ, Λ12)
end

function condmean_prec( x::Vector{<:AbstractFloat},
                   mu::Vector{<:AbstractFloat},
                   Σ::Matrix{<:AbstractFloat},
                   Λ12::Matrix{<:AbstractFloat},
                   jinds::UnitRange{Int},
                   notjinds::Vector{Int} )
    
    #estimate conditional mu
    μ = mu[jinds] - Σ*Λ12*(x[notjinds] - mu[notjinds])
    
    return μ
end

function cond_mean_prec( x::Vector{<:AbstractFloat},
                        mu::Vector{<:AbstractFloat},
                        S::Matrix{<:AbstractFloat},
                        jinds::UnitRange{Int},
                        notjinds::Vector{Int} )
    
    # Σ[j|notj]
    (Σ, Λ12) = condcov_prec(S, jinds, notjinds)
    
    # μ[j|notj]
    μ = condmean_prec(x, mu, Σ, Λ12, jinds, notjinds)
    
    return (μ, Σ)
end

function MvNormal_Gibbs_Precision( mu::Vector{<:AbstractFloat},
                            S::Matrix{<:AbstractFloat},
                            x::Vector{<:AbstractFloat},
                            nsub::Int )
    
    #Check nsub
    nfull   = length(mu)::Int
    sub_len = div(nfull, nsub)::Int
    
    for j in 1:nsub #number of subvector
        
        # set j-th block indices
        (jinds, notjinds) = setjinds(j, sub_len, nfull)
        
        # compute conditional mean and covariance
        (μ, Σ) = cond_mean_prec(x, mu, S, jinds, notjinds)
        
        #sample conditional vector
        x[jinds] = rand(MvNormal(μ, Σ))
        
    end
    
    return x
end

# multiple iterations
function MvNormal_Gibbs_Precision( nsim::Int,
                         mu::Vector{<:AbstractFloat},
                         S::Matrix{<:AbstractFloat},
                         nsub::Int )
    
    #preallocate
    samp      = ones( length(mu), nsim )
    samp[:,1] = mu
    
    #Gibbs sample multiple draws
    for i in 2:nsim
        
        samp[:,i] = MvNormal_Gibbs_Precision(mu, S, samp[:,i-1], nsub)
        
    end
    
    return samp
end
