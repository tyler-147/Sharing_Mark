#=Joint Distribution Test from Geweke (2004) using the first and second moments
as test functions. Uses overlapping batch means(OBM) (Meketon and Schmeiser, 2007) to
estimate the asymptotic variance of the successive-conditional simulation. =#

#load packages
using Distributions, Statistics, Gadfly

# -----------------------------------------------------------------------------
#Simulation Settings
# -----------------------------------------------------------------------------

nsim = 500             #number of simulations
burn = 251             #number of sumulations burned
dim  = nsim - burn + 1 #number of simulations left after burning
bins = 20              #size of bins used in OBM

# -----------------------------------------------------------------------------
#Required Functions
# -----------------------------------------------------------------------------

function first_second_moment(data)
#= Calculates the first and second moments of the provided vector
  data  = vector that function will be played on
  returns [1 x 2] vector of the first moment (first_mom) and
  second moment (second_mom)=#

  lngth = length(data)

  first_mom  = mean(data)
  second_mom = var(data)

  return first_mom, second_mom
end

function variance_test_function(data, result)
  #= Estimates the varaince of the test function results, giving the estimated
    variance for both the first and second moments
    data   = the data used to estimate the result of the test function
    result = the result of the test function applied to the test function
    moment = which moment that the test function is
    Results a [1 x 2] that reports the variance of the first moment (fvar)
    and the variance of the second moment (svar) =#

    input_1 = data.^2 .- result[1]^2
    fvar    = 1/length(data)*sum(input_1)

    input_2 = (data .- mean(data)).^4 .- result[2]^2
    svar    = 1/length(data)*sum(input_2)

    return fvar, svar
end

function step_ahead(beta, variance, Y0)
  #= Given a beta,error variance, and initial value, simulate an AR(1) with 5
     datapoints.
     beta     = the coefficient of previous data value
     variance = the variance of the error
     Y0       = the initial value
     Returns a (1 x 5) vector (simul_data) of simulated data, starting at Y0 =#

     simul_data    = ones(201)
     simul_data[1] = Y0

     for i in 2:201
       simul_data[i] = simul_data[i-1]*mean_pri + rand(Normal(0,simul_data[i]))
     end

     simul_data = simul_data[2:end]

     return simul_data
end

function batch_variance(data, batch)
  #= Estsimate the asymptotic variance using overlapping batch means
     data = the data that the variance of is desired
     batch = the size of the batches that the data will be quartered into
     Returns a scalar estiamte (batch_var) of the asymptotic variance=#

  #Setup
  lng   = length(data)
  means = ones((lng - batch))

  #Take a rolling average with length of window equal to batch
  for i in 1:(lng - batch)
    means[i] = mean(data[i:i+batch])
  end

  #Estimate
  batch_var = sum((means .- mean(data)).^2)*lng*batch/((lng - batch)*(lng - batch + 1))
end
# -----------------------------------------------------------------------------
#Prior Specifications
# -----------------------------------------------------------------------------

# Y |β,σ²,Y⁰ ~ Normal( β*Y⁰ , σ²)

##Conjugate NIG
# σ²      ~ InverseGamma( shape , scale   )
# β |σ²   ~ Normal( mean , σ²*inv*(h) )

#Independent NIG
# σ² ~ InverseGamma( shape , scale   )
# β  ~ Normal( mean , σ²*inv*(h) )

#Set up hyperparameters
mean_pri  = 0.2
var_pri   = 10
shape_pri = 7
scale_pri = 5
Y0        = .5

# -----------------------------------------------------------------------------
#Marginal Conditional Simulator - Conjugate NIG
# -----------------------------------------------------------------------------

#Sample parameters from prior distribtion
mparam_conj      = ones(nsim,2)
mparam_conj[:,2] = scale_pri ./ rand(Chisq(shape_pri),nsim)

for i in 1:nsim
  mparam_conj[i,1] = rand(Normal(mean_pri,inv(var_pri)*mparam_conj[i,2]))
end

#Sample data conditional on each set of parameters
mdata_conj = ones(200,nsim)

for i in 1:nsim
  mdata_conj[:,i] = step_ahead(mparam_conj[i,1], mparam_conj[i,2], Y0)
end

#Burn initial estiamtes
mparam_conj = mparam_conj[burn:end,:]

#Estimate test function and variance of test function for each parameter
mbeta_mom_conj     = first_second_moment(mparam_conj[:,1])
msigma_sq_mom_conj = first_second_moment(mparam_conj[:,2])

#Estimate the variance of each test function
mbeta_var_conj     = variance_test_function(mparam_conj[:,1], mbeta_mom_conj)
msigma_sq_var_conj = variance_test_function(mparam_conj[:,2], msigma_sq_mom_conj)

# -----------------------------------------------------------------------------
# Successive-Conditional Simulator (Conjugate)
# -----------------------------------------------------------------------------

#Sample initial parameters from prior
int_sigma_sq = scale_pri ./rand(Chisq(shape_pri))
int_beta     = rand(Normal(mean_pri,inv(var_pri)*int_sigma_sq))
int_param    = [int_beta, int_sigma_sq]

#Simulate data and parameters with Gibbs Sampler
sparam_conj      = ones(nsim, 2)
sparam_conj[1,:] = int_param
sdata_conj       = ones(200, nsim)

for i in 1:nsim
  #Simulate data conditional on previous parameters
  sdata_conj[:,i] = step_ahead(sparam_conj[i,1], sparam_conj[i,2], Y0)

  #Simulate parameters conditional on previous data
  if i!=nsim
    #Estimate OLS and get SE
    Y_1       = pushfirst!(sdata_conj[1:end-1,i], Y0)
    Y         = sdata_conj[1:end,i]
    beta_OLS  = inv(Y_1'*Y_1)*Y_1'*Y
    errors_sq = (Y - Y_1*beta_OLS)'*(Y - Y_1*beta_OLS)

    #Update parameters
    var_post   = var_pri + Y_1'*Y_1
    mean_post  = inv(var_post)*(var_pri*mean_pri + Y_1'*Y_1*beta_OLS)
    shape_post = shape_pri + length(Y)
    scale_post = errors_sq + mean_pri'*var_pri*mean_pri + beta_OLS'*Y_1'*Y_1*beta_OLS - mean_post'*var_post*mean_post

    #Sample sigma squared and then beta conditional on sigma squared
    sparam_conj[i+1,2] = scale_post ./rand(Chisq(shape_post))
    sparam_conj[i+1,1] = rand(Normal(mean_post,inv(var_post)*sparam_conj[i+1,2]))
  end
end

#Burn initial observations
sparam_conj = sparam_conj[burn:end,:]

#Estimate test functions each parameter
sbeta_mom_conj     = first_second_moment(sparam_conj[:,1])
ssigma_sq_mom_conj = first_second_moment(sparam_conj[:,2])

#Use OBM to estimate the variance of first moment
sbeta_var_conj        = ones(2)
ssigma_sq_var_conj    = ones(2)
sbeta_var_conj[1]     = batch_variance(sparam_conj[:,1],bins)
ssigma_sq_var_conj[1] = batch_variance(sparam_conj[:,2],bins)

#Use OBM to estimate variance of second moment
sec_beta              = (sparam_conj[:,1] .- mean(sparam_conj[:,2])).^2
sec_sig_sq            = (sparam_conj[:,2] .- mean(sparam_conj[:,2])).^2
sbeta_var_conj[2]     = batch_variance(sec_beta,bins)
ssigma_sq_var_conj[2] = batch_variance(sec_sig_sq,bins)

# -----------------------------------------------------------------------------
# Successive-Conditional Simulation (Independent)
# -----------------------------------------------------------------------------

#Sample initial parameters from prior
int_sigma_sq = scale_pri ./rand(Chisq(shape_pri))
int_beta     = rand(Normal(mean_pri,inv(var_pri)*int_sigma_sq))
int_param    = [int_beta, int_sigma_sq]

#Parameter update
shape_post_ind = shape_pri + 200

#Fill in
sparam_ind      = ones(nsim, 2)
sparam_ind[1,:] = int_param
sdata_ind       = ones(200, nsim)

for i in 1:nsim
  #Simulate data conditional on previous data
  sdata_ind[:,i] = step_ahead(sparam_ind[i,1], sparam_ind[i,2], Y0)

  #Simulate parameters
  if i!=nsim
    #Estimate OLS and get SE
    Y_1 = pushfirst!(sdata_conj[1:end-1,i], Y0)
    Y   = sdata_conj[1:end,i]

    #Update parameters
    prec_ind = var_pri + sparam_ind[i,2]*Y_1'*Y_1
    mean_ind = inv(prec_ind)*(var_pri*mean_pri + sparam_ind[i,2]*Y_1'*Y)

    #Sample beta
    sparam_ind[i+1,1] = rand(Normal(mean_ind, inv(prec_ind)))

    #Sample sigma squared
    scale_ind         = scale_pri + (Y - Y_1*sparam_ind[i+1,1])'*(Y - Y_1*sparam_ind[i+1,1])
    sparam_ind[i+1,2] = scale_ind / rand(Chisq(shape_post_ind))
  end
end

#Burn intitial estimates
sparam_ind = sparam_ind[burn:end,:]
sdata_indb = sdata_ind[:,burn:end]

#Estimate test functions
sbeta_mom_ind     = first_second_moment(sparam_ind[:,1])
ssigma_sq_mom_ind = first_second_moment(sparam_ind[:,2])

#Use OBM to estimate the variance of first moment
sbeta_var_ind        = ones(2)
ssigma_sq_var_ind    = ones(2)
sbeta_var_ind[1]     = batch_variance(sparam_ind[:,1],bins)
ssigma_sq_var_ind[1] = batch_variance(sparam_ind[:,2],bins)

#Use OBM to estimate variance of second moment
sec_beta_ind         = (sparam_ind[:,1] .- mean(sparam_ind[:,2])).^2
sec_sig_sq_ind       = (sparam_ind[:,2] .- mean(sparam_ind[:,2])).^2
sbeta_var_ind[2]     = batch_variance(sec_beta_ind,bins)
ssigma_sq_var_ind[2] = batch_variance(sec_sig_sq_ind,bins)

# -----------------------------------------------------------------------------
# Estimate Geweke test statistic
# -----------------------------------------------------------------------------

beta_first_conj    = (mbeta_mom_conj[1] - sbeta_mom_conj[1])/(1/dim*(mbeta_var_conj[1] + sbeta_var_conj[1]))^(1/2)
beta_second_conj   = (mbeta_mom_conj[2] - sbeta_mom_conj[2])/(1/dim*(mbeta_var_conj[2] + sbeta_var_conj[2]))^(1/2)
sig_sq_first_conj  = (msigma_sq_mom_conj[1] - ssigma_sq_mom_conj[1])/(1/dim*(msigma_sq_var_conj[1] + ssigma_sq_var_conj[1]))^(1/2)
sig_sq_second_conj = (msigma_sq_mom_conj[2] - ssigma_sq_mom_conj[2])/(1/dim*(msigma_sq_var_conj[2] + ssigma_sq_var_conj[2]))^(1/2)

beta_first_ind    = (mbeta_mom_conj[1] - sbeta_mom_ind[1])/(1/dim*(mbeta_var_conj[1] + sbeta_var_ind[1]))^(1/2)
beta_second_ind   = (mbeta_mom_conj[2] - sbeta_mom_ind[2])/(1/dim*(mbeta_var_conj[2] + sbeta_var_ind[2]))^(1/2)
sig_sq_first_ind  = (msigma_sq_mom_conj[1] - ssigma_sq_mom_ind[1])/(1/dim*(msigma_sq_var_conj[1] + ssigma_sq_var_ind[1]))^(1/2)
sig_sq_second_ind = (msigma_sq_mom_conj[2] - ssigma_sq_mom_ind[2])/(1/dim*(msigma_sq_var_conj[2] + ssigma_sq_var_ind[2]))^(1/2)
