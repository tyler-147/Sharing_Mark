# Multivariate normal test (Mardia, 1970)

#load packages
using Statistics, Distributions

# -----------------------------------------------------------------------------
# Function
# -----------------------------------------------------------------------------

function multivar_test(input, p_value)
    #= Uses Mardia's test to check if a given vector of data is multivariate
    normal distribution.
    input   = n  k-dimensional vectors (n x p) that will be tested to see if it is multivariate normal
    p_value = the p-value level that constitutes rejecting the null hypothesis
    Returns a (3 x 1) vector (result), with result[1] representing the skewness
    and the kurtosis first. 1  it if passes test (is
    multivariate normal) and 0 if fails (is not multivariate normal). result[2]
    is the 1 - p-value of the skewness test statistic, and result[3] is the
    1 - p-value of the kurtosis test statistic. =#

    #Dimensions of the dataset
    dims = size(input)

    #Mean of draws
    m_draw = mean(input, dims = 2)

    #Covariance of draws without correcting for degrees of freedom
    sigma     = cov(input', corrected = false)
    inv_sigma = inv(sigma)

    #Estimate multivariate skewness
    fill = ones(dims[2], dims[2])

    for i in 1:dims[2]
        for j in 1:dims[2]
            am        = 1/(dims[2]^2)*((input[:,i] - m_draw)'*inv_sigma*(input[:,j] - m_draw))^3
            fill[i,j] = am[1]
        end
    end

    skew = sum(fill)

    #Estimate multivariate kurtosis
    fill = ones(dims[2])

    for i in 1:dims[2]
        am      = 1/dims[2]*((input[:,i] - m_draw)'*inv_sigma*(input[:,i] - m_draw))^2
        fill[i] = am[1]
    end

    kurt = sum(fill)

    #Find p-value for skewness
    A           = dims[2]/6*skew
    deg_free    = dims[1]*(dims[1] + 1)*(dims[1] + 2)/6
    skew_result = 1 - cdf(Chisq(deg_free), A)

    #Find p-value for kurtosis
    B           = (kurt - dims[1]*(dims[1] + 2))*sqrt(dims[2]/(8*dims[1]*(dims[1] + 2)))
    kurt_result = 1 - cdf(Normal(0,1), abs(B))

    #Compare the reults to the desired p-value
    result = zeros(2)

    if skew_result >= p_value
        result[1] = 1
    end

    if kurt_result >= p_value/2
        result[2] = 1
    end

    return result, skew_result, kurt_result
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
sample_e4 = rand(MvTDist(4, zeros(n_vec), cov_draw), n_draws)

#Prep for matrix t-distribution

#Check results
res    = multivar_test(sample, p_value)
e1_res = multivar_test(sample_e1, p_value)
e2_res = multivar_test(sample_e2, p_value)
e3_res = multivar_test(sample_e3, p_value)
e4_res = multivar_test(sample_e4, p_value)
