#= Joint Distribution Test of an AR(1) model following Geweke (2004). The test 
statistics presented at the end of the script are approximately distributed by a 
unit normal distribution. If the test statistic is outside of the desired critial 
value, the test fails, and there is some error in the theory or code. If the test
statistic is inside the critical values, this test suggests that the posterior 
simulation and prior distribution are of the same joint distribution. =#

#load packages
using Gadfly, Statistics, Distributions

# -----------------------------------------------------------------------------
# Required functions
# -----------------------------------------------------------------------------

function step_ahead(beta, variance, Y0, nsim, steps)
    #= Estimate a step ahead for an AR(1)
       beta     = the coefficient of the lagged variable
       variance = the variance of the error term
       Y0       = starting value
       nsim     = number of times simulation run
       steps    = number of steps ahead the forecast is run
       Returns a (steps x nsim) matrix of forecasts =#

    #Create empty matrix to fill in
    fill      = ones(steps, nsim)
    fill[1,:] = fill[1,:]*Y0

    #Forecast ahead one step at a time
    for i in 1: nsim
        for j in 2:steps
            fill[j,i] = beta*fill[j-1,i] + rand(Normal(0,sqrt(variance)))
        end
    end

    return fill
end

function marginal_conditional(Y0, shape, scale, mean, var, nsim, tsim)
    #= Draws sample from joint distribution of parameters and data using
    marginal-conditional simulator with Normal Inverse-Gamma prior
    Y0    = initial data starting point
    shape = IG shape prior for variance parameter
    scale = IG scale prior for variance parameter
    mean  = Normal mean prior for beta parameter
    var   = Normal variance prior for beta parameter
    nsim  = number of samples desired from joint distribution
    tsim  = number of steps ahead desired for drawn data
    Returns a [3] vector of draws from joint distribution, where each element is
    either a vector or matrix =#

    #Draw variance parameters from prior distribution
    sig_sq = rand(InverseGamma(shape, scale), nsim)

    #Draw betas conditional on previously drawn variance parameters
    betas = ones(nsim)

    for i in 1:nsim
        betas[i] = rand(Normal(mean, sqrt(inv(var)*sig_sq[i])))
    end

    #Draw data conditional on parameters
    data = ones(tsim, nsim)

    for i in 1:nsim
        data[:,i] = step_ahead(betas[i], sig_sq[i], Y0, 1, tsim)
    end

    return betas, sig_sq, data
end

function marginal_conditional_var(draws)
    #= This function estimates the variance of draws using a marginal-conditional
    simulation.
    draws = draws using marginal-conditional simulator with some test function
    Returns a scalar (variance) estimate of the variance. =#

    #Size of draws
    n = length(draws)

    #Mean of draws
    draw_mean = mean(draws)

    #Estimate variance
    variance = 0
    for i in 1:n
        variance = variance + inv(n)*(draws[i]^2 - draw_mean^2)
    end

    return variance
end

function successive_posterior(Y0, shape, scale, mean, var, nsim, tsim)
    #= Draws sample from joint distribution of parameters and data using
    successive-posterior simulator with Normal Inverse-Gamma prior with parameters
    updated in line with conjuagte NIG prior.
    Y0    = initial data starting point
    shape = IG shape prior for variance parameter
    scale = IG scale prior for variance parameter
    mean  = Normal mean prior for beta parameter
    var   = Normal variance prior for beta parameter
    nsim  = number of samples desired from joint distribution
    tsim  = number of steps ahead desired for drawn data
    Returns a [3] vector of draws from joint distribution, where each element is
    either a vector or matrix =#

    #Draw inital sample of parameters from prior distribution
    sig_pri  = rand(InverseGamma(shape, scale))
    beta_pri = rand(Normal(mean, sqrt(inv(var)*sig_pri)))

    #Create vectors to fill in with sampled data
    sig_sq = ones(nsim)
    beta   = ones(nsim)
    data   = ones(tsim, nsim)

    #Successively draw data and parameters conditional on the other
    for i in 1:nsim
        #Draw data conditional on parameters
        if i == 1
            data[:,i] = step_ahead(beta_pri, sig_pri, Y0, 1, tsim)
        else
            data[:,i] = step_ahead(beta[i - 1], sig_sq[i-1], Y0, 1, tsim)
        end

        #Organize data as AR(1) model
        Y   = reverse(data[2:end,i])
        Y_1 = reverse(data[1:end-1,i])

        #Estimate OLS betas
        beta_OLS = inv(Y_1'*Y_1)*Y_1'*Y

        #Update parameter distribution hyperparameters using drawn data
        var_post   = Y_1'*Y_1 + var
        mean_post  = inv(var_post)*(var*mean + Y_1'*Y_1*beta_OLS)
        shape_post = shape + (tsim - 1)/2
        scale_post = scale + .5*(Y'*Y + mean'*var*mean - mean_post'*var_post*mean_post)

        #Draw parameters conditional on data with updated parameters
        sig_sq[i] = rand(InverseGamma(shape_post, scale_post))
        beta[i]   = rand(Normal(mean_post, sqrt(inv(var_post)*sig_sq[i])))

    end
    return beta, sig_sq, data
end

function numerical_SE(draws)
    #= Estimates the numerical standard error of an MCMC given transformed
    draws from a given distribution.
    draws = transformed draws from a given distribution
    Returns a scalar (tao) estimate of the numerical SE of given draws =#

    #Size of draws
    n = length(draws)

    #Mean of draws
    draw_mean = mean(draws)

    #Loading function
    L = L = Int(ceil(n^(1 / 2.1)))

    #Autocovariance adjustment
    auto =  ones(n)

    for i in 0:n - 1
        fill = 0
        for j in i+1:n
            fill = fill + (draws[j] - draw_mean)*(draws[j - i] - draw_mean)
        end
        auto[i + 1] = inv(n)*fill
    end

    #SE estimation
    full = ones(L - 1)

    for i in 1:L - 1
        full[i] = (L - i)/L*auto[i + 1]
    end

    tao = auto[1] + 2*sum(full)

    return tao
end

function geweke_test_first_mom(marg, succ, burn)
    #= This function estimates the geweke test statistic for draws from a
    marginal-conditional and successive-conditional simulators using the first
    moment of the data as the test function.
    marg = vector of draws from a marginal-conditional simulator
    succ = vector of draws from a successive-conditional simulato
    burn = number of initial simulations that are being burned
    Returns the test statistic (geweke_test) of geweke test statistic
    t (t ~ Normal(0,1)). =#

    #Burn inital estimates
    marg = marg[burn:end]
    succ = succ[burn:end]

    #Length of draws vectors
    n_marg = length(marg)
    n_succ = length(succ)

    #Estimate mean of draws
    marg_mean = mean(marg)
    succ_mean = mean(succ)

    #Estimate variance of each set of draws
    var_marg = marginal_conditional_var(marg)
    var_succ = numerical_SE(succ)

    #Estimate test statistic
    geweke_test = (marg_mean - succ_mean)/sqrt(inv(n_marg)*var_marg + inv(n_succ)*var_succ)

    return geweke_test
end

function geweke_test_second_mom(marg, succ, burn)
    #= This function estimates the geweke test statistic for draws from a
    marginal-conditional and successive-conditional simulators using the second
    moment of the data as the test function.
    marg = vector of draws from a marginal-conditional simulator
    succ = vector of draws from a successive-conditional simulato
    burn = number of initial simulations that are being burned
    Returns the test statistic (geweke_test) of geweke test statistic
    t (t ~ Normal(0,1)). =#

    #Burn inital estimates
    marg = marg[burn:end]
    succ = succ[burn:end]

    #Length of draws vectors
    n_marg = length(marg)
    n_succ = length(succ)

    #Apply second moment as test function
    marg_sec = (marg .- mean(marg)).^2
    succ_sec = (succ .- mean(succ)).^2

    #Estimate mean of draws post function applied
    marg_mean = mean(marg_sec)
    succ_mean = mean(succ_sec)

    #Estimate variance of each set of draws
    var_marg = marginal_conditional_var(marg_sec)
    var_succ = numerical_SE(succ_sec)

    #Estimate test statistic
    geweke_test = (marg_mean - succ_mean)/sqrt(inv(n_marg)*var_marg + inv(n_succ)*var_succ)

    return geweke_test
end

# -----------------------------------------------------------------------------
# Simulation settings
# -----------------------------------------------------------------------------

Y0        = 1
shape_pri = 10
scale_pri = 20
mean_pri  = .4
var_pri   = 30
nsim      = 100000
tsim      = 5
burn      = 30001

# -----------------------------------------------------------------------------
# Draw samples and estimate test statistic
# -----------------------------------------------------------------------------

#Sample betas and sigma-sq from marginal-conditional and successive-conditonal simulators
a = marginal_conditional(Y0, shape_pri, scale_pri, mean_pri, var_pri, nsim, tsim)
b = successive_posterior(Y0, shape_pri, scale_pri, mean_pri, var_pri, nsim, tsim)

#Test betas
geweke_beta_first  = geweke_test_first_mom(a[1], b[1], burn)
geweke_beta_second = geweke_test_second_mom(a[1], b[1], burn)

#Test sig_sq
geweke_sig_sq_first   = geweke_test_first_mom(a[2], b[2], burn)
geweke_sig_sq2_second = geweke_test_second_mom(a[2], b[2], burn)
