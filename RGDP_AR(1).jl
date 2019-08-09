#AR(1) bayesian regression of RGDP growth rates with plots

#load packages
using CSV, DataFrames, Statistics, Distributions, Gadfly

# -----------------------------------------------------------------------------
#Required Functions
# -----------------------------------------------------------------------------

function sample_NIG(nsim, IG_shape, IG_scale, N_mean, N_var)
#= This function samples σ² from an Inverse-Gamma distribution given scale and shape 
  parameters, and then draws from a normal distribution conditional on the drawn 
  σ². 
  nsim    = number of samples desired
  IG_shape = shape parameter for IG distribution
  IG_scale = scale parameter fro IG distribution
  N_mean   = mean parameter for Normal distribution
  N_var    = variance parameter for Normal distribution 

  Returns an (nsamp x 2) arrary [σ² betas] =#

  #Sample σ² 
  sig_sq = rand(InverseGamma(IG_shape, IG_scale),nsim)

  #Sample a beta conditional on each sig_sq
  betas = ones(nsim)
  
  for i in 1:nsim
    betas[i,] = rand(Normal(N_mean,inv(N_var)*sig_sq[i,1]))
  end

  return sig_sq, betas
end


# -----------------------------------------------------------------------------
# Simulation settings
# -----------------------------------------------------------------------------

nsim = 500

# -----------------------------------------------------------------------------
# Prior
# -----------------------------------------------------------------------------

# σ²        ~ InverseGamma( shape, scale   )
# beta | σsq ~ Normal( mean , σ²*inv*(h) )

#Set up hyperparameters
mean_pri  = 0.5
var_pri   = 10
shape_pri = 7
scale_pri = 5

# -----------------------------------------------------------------------------
# File I/O settings
# -----------------------------------------------------------------------------

if homedir() == "C:\\Users\\tgwin"

  location_data = "C:\\Users\\tgwin\\OneDrive\\Fed\\Mark Bognanni"

elseif homedir() == "C:\\Users\\Mark"

  location_data = "C:\\Users\\Mark\\Documents\\git_repos\\Sharing_Mark"

end

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

#load RGDP growth FRED data from csv
cd(location_data)

frame       = CSV.read("RGDP_Practice.csv")
frame_arr   = convert(Matrix, frame)
RGDP        = convert(Array{Float64}, frame_arr[:,2])
demean_RGDP = RGDP .- mean(RGDP)

#Set up lagged data
Y   = demean_RGDP[2:end]
Y_1 = demean_RGDP[1:end-1]

# -----------------------------------------------------------------------------
# Posterior
# -----------------------------------------------------------------------------

#Estimate OLS betas and get SSE
beta_OLS  = inv(Y_1'*Y_1)*Y_1'*Y
errors_sq = (Y - Y_1*beta_OLS)'*(Y - Y_1*beta_OLS)

#Update parameters
var_post   = var_pri + Y_1'*Y_1
mean_post  = inv(var_post)*(var_pri*mean_pri + Y_1'*Y_1*beta_OLS)
shape_post = shape_pri + length(Y)/2
scale_post = scale_pri + 1/2*(Y'*Y + mean_pri*var_pri*mean_pri - mean_post*var_post*mean_post)

#Take nsim samples of sigma^2 and betas from posterior and prior distributions
(sig_sq, betas)   = sample_NIG(nsim, shape_post, scale_post , mean_post, var_post)
(sig_sq_pri, betas_pri) = sample_NIG(nsim, shape_pri, scale_pri ,mean_pri, var_pri)


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

#Prior and Posterior Densities of Beta
p1 = plot(layer(x = betas_pri, Geom.density),
          layer(x = betas, Geom.density, Theme(default_color = "green")),
          Guide.ylabel("Density"),Guide.title("Prior and Posterior Density of β"),
          Coord.cartesian(xmin = .2, xmax = .7, ymin = 0, ymax = 80),
          Guide.manual_color_key("",["Prior","Posterior"],[Gadfly.current_theme().default_color,"green"]))

#Prior and Posterior Densities of Sigma Squared
p2 = plot(layer(x = sig_sq_pri, Geom.density),
          layer(x = sig_sq, Geom.density, Theme(default_color = "green")),
          Guide.ylabel("Density"),Guide.title("Prior and Posterior Density of σ²"),
          Coord.cartesian(xmin = 0, xmax = 13, ymin = 0, ymax = 1.5),
          Guide.manual_color_key("",["Prior","Posterior"],[Gadfly.current_theme().default_color,"green"]))

#Contour plot of Normal-Gamma joint prior distribution and
p3 = plot(layer(z=(x,y) -> 1/(factorial(shape_pri-1)/mean_pri^shape_pri*(2*pi/var_pri)^(1/2))*y^(shape_pri - 1/2)*exp((-1)*y/2*(var_pri*(x - mean_pri)^2 + 2*mean_pri)),
            xmin=[0], xmax=[1], ymin=[2], ymax=[30], Geom.contour),
          layer(x = betas, y = sig_sq, Geom.point),
          Guide.ylabel("σ²"), Guide.xlabel("β"),
          Guide.title("Joint Prior Normal-Gamma Distribution Contour with Sampled Posterior Parameters"))

display(p1)
display(p2)
display(p3)
