# Multivariate normal test (Mardia, 1970)

#load packages
using Statistics, Distributions
using Random

Random.seed!(8675309)

# -----------------------------------------------------------------------------
# Function
# -----------------------------------------------------------------------------

function multivar_test(input::Matrix{<: AbstractFloat}, p_value::AbstractFloat)
    #= Uses Mardia's test to check if a given vector of data is multivariate
    normal distribution.
    input   = n  p-dimensional vectors (p x n) that will be tested to see if it is multivariate normal
    p_value = the p-value level that constitutes rejecting the null hypothesis
    Returns a (3 x 1) vector (result), with result[1] representing the skewness
    and the kurtosis first. 1  it if passes test (is
    multivariate normal) and 0 if fails (is not multivariate normal). result[2]
    is the 1 - p-value of the skewness test statistic, and result[3] is the
    1 - p-value of the kurtosis test statistic. =#

    #Dimensions of the dataset
    dims = size(input)

    p, ndraws = dims[1], dims[2]

    #Mean of draws
    m_draw = mean(input, dims = 2)[:] 
    # trailing [:] operation makes it a Vector in the Julia sense of having second dimension equal to 1

    #Covariance of draws without correcting for degrees of freedom
    sigma     = cov(input', corrected = false)
    inv_sigma = inv(sigma)

    # ---------------------------------
    #Estimate multivariate skewness
    # ---------------------------------

    b1p = 0.0

    # Mardia (1974) 2.2
    for i in 1:ndraws

        xim_invS = (input[:,i] - m_draw)' * inv_sigma

        for j in 1:ndraws
            
            b1p += ( xim_invS * (input[:,j] - m_draw) )^3
        
        end
    end

    b1p = (1/ndraws^2) * b1p

    # ---------------------------------
    #Estimate multivariate kurtosis
    # ---------------------------------

    b2p = 0.0 # sample kurtosis

    # implement Mardia (1970) 3.12
    for i in 1:ndraws
        
        xim  = input[:,i] - m_draw
        b2p += ( xim' * inv_sigma * xim )^2
    
    end

    b2p = (1/ndraws) * b2p

    # ---------------------------------
    # p values for test statistics
    # ---------------------------------

    #Find p-value for skewness
    A               = (ndraws/6)*b1p # Mardia (1974) 5.4
    deg_free        = p*(p + 1)*(p + 2)/6
    skew_test_pval  = 1 - cdf(Chisq(deg_free), A)

    #Find p-value for kurtosis
    # beta2p : true value of multi. kurt. under Mardia's def
    beta2p          = p*(p+2) # Mardia (1970) 3.11
    B               = (b2p - beta2p) / sqrt( (8*p*(p + 2))/ndraws )
    kurt_test_pval  = 1 - cdf(Normal(0,1), abs(B))

    #Compare the reults to the desired p-value
    result = zeros(2)

    if skew_test_pval >= p_value
        result[1] = 1
    end

    if kurt_test_pval >= p_value/2
        result[2] = 1
    end

    return result, skew_test_pval, kurt_test_pval, b1p, b2p
end

# -----------------------------------------------------------------------------
# Example
 # ----------------------------------------------------------------------------

#Specifications
n_vec     = 5
n_draws   = 5000
mean_draw = rand(n_vec)
fill      = rand(n_vec, n_vec)
cov_draw  = fill*fill'
p_value   = 0.05

#Sample from multivariate-normal with specifications
sample = rand(MvNormal(mean_draw, cov_draw), n_draws)

#Sample from non-MVnormal distributions
sample_e1 = rand(MvLogNormal(mean_draw, cov_draw), n_draws)
sample_e2 = rand(TDist(5), (n_vec, n_draws))
sample_e3 = rand(Uniform(-50, 50), (n_vec, n_draws))
sample_e4 = rand(MvTDist(5, zeros(n_vec), cov_draw), n_draws)

print("samples drawn")

#Check results
res    = multivar_test(sample, p_value)
e1_res = multivar_test(sample_e1, p_value)
e2_res = multivar_test(sample_e2, p_value)
e3_res = multivar_test(sample_e3, p_value)
e4_res = multivar_test(sample_e4, p_value)
