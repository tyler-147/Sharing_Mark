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

function MvNormal_Gibbs(nsim::Integer, mean_vector::Array{Float64,1},
                        covariance::Array{Float64,2}, nsub::Integer)

    #length
    len = length(mean_vector)

    #Check nsub
    check   = len/nsub

    if check != round(check, digits = 0)

        throw(UndefVarError(:nsubdoesnotdividevectorevenly))

    else

        sub_len = convert(Integer, check)

    end

    #Set up matrix for sampling
    samp      = ones(len, nsim)
    samp[:,1] = mean_vector

    #Gibbs sample multivariate
    for i in 2:nsim
        for j in 1:nsub

            #reference indices
            start = (j - 1)*sub_len + 1
            stop  = sub_len*j

            #Form vector of conditional
            a = vcat(samp[1:start - 1,i], samp[stop + 1:end, i - 1])

            #Σ11
            Σ11 = covariance[start:stop , start:stop]

            #Σ12
            Σ12 = hcat(covariance[start:stop, 1:start - 1],
                       covariance[start:stop, stop+1:end])

            #Σ21
            Σ21 = vcat(covariance[1:start - 1, start:stop],
                       covariance[stop + 1:end, start:stop])

            #Σ22
            cov1 = vcat(covariance[1:start - 1, 1:start - 1],
                        covariance[stop + 1:end, 1:start - 1])
            cov2 = vcat(covariance[1:start - 1, stop + 1:end],
                        covariance[stop + 1:end, stop + 1:end])
            Σ22  = hcat(cov1, cov2)

            #μ
            μ1 = mean_vector[start:stop]
            μ2 = vcat(mean_vector[1:start - 1], mean_vector[stop + 1:end])

            #Mean for sampling distribution
            μ = μ1 + Σ12*inv(Σ22)*(a - μ2)

            #Covariance for sampling distribution
            Σ = Σ11 - Σ12*inv(Σ22)*Σ21
            Σ = 0.5*(Σ + Σ')

            #Sample
            samp[start:stop, i] = rand(MvNormal(μ, Σ))

        end
    end

    return samp

end

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------

nsim        = 1000
vec_len     = 20
mean_vector = rand(vec_len)
prep        = rand(vec_len,vec_len)
covariance  = (prep + prep')/2 + vec_len*Matrix{Float64}(I, vec_len, vec_len )
nsub        = 5
burn        = 300

# -----------------------------------------------------------------------------
# Example
# -----------------------------------------------------------------------------

samp = MvNormal_Gibbs(nsim, mean_vector, covariance, nsub)

#Check difference in mean
mean_check = mean(samp[:,burn:end] , dims = 2) - mean_vector
