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
  nsim     = number of samples desired
  IG_shape = shape parameter for IG distribution
  IG_scale = scale parameter fro IG distribution
  N_mean   = mean parameter for Normal distribution
  N_var    = variance parameter for Normal distribution
  Returns an (nsamp x 2) arrary [σ² betas] =#

  #Sample σ²
  sig_sq = 1 ./(rand(Chisq(IG_shape),nsim) ./ IG_scale)

  #Sample a beta conditional on each sig_sq
  betas = ones(nsim)

  for i in 1:nsim
    betas[i,] = rand(Normal(N_mean,sqrt(inv(N_var)*sig_sq[i,1])))
  end

  return sig_sq, betas
end

# -----------------------------------------------------------------------------
# Simulation settings
# -----------------------------------------------------------------------------

nsim = 5000

# -----------------------------------------------------------------------------
# Prior Specifications
# -----------------------------------------------------------------------------

#Conjugate NIG
# σ²        ~ InverseGamma( shape, scale   )
# β |σ²   ~ Normal( mean , σ²*inv*(h) )

#Independent NIG
# σ²        ~ InverseGamma( shape, scale   )
# β  ~ Normal( mean , σ²*inv*(h) )

#Set up hyperparameters
mean_pri  = 0.5
var_pri   = 5
shape_pri = 7
scale_pri = 5

# -----------------------------------------------------------------------------
# File I/O settings
# -----------------------------------------------------------------------------

if homedir() == "C:\\Users\\tgwin"

  location_data = "C:\\Users\\tgwin\\OneDrive\\Fed\\Mark Bognanni\\Practice"

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
# Conjugate Normal Inverse-Gamma Posterior
# -----------------------------------------------------------------------------

#Estimate OLS betas and get SSE
beta_OLS  = inv(Y_1'*Y_1)*Y_1'*Y
errors_sq = (Y - Y_1*beta_OLS)'*(Y - Y_1*beta_OLS)

#Update parameters
var_post   = var_pri + Y_1'*Y_1
mean_post  = inv(var_post)*(var_pri*mean_pri + Y_1'*Y_1*beta_OLS)
shape_post = shape_pri + length(Y)
scale_post = errors_sq + mean_pri'*var_pri*mean_pri + beta_OLS'*Y_1'*Y_1*beta_OLS - mean_post'*var_post*mean_post

#Take nsim samples of sigma^2 and betas from posterior and prior distributions
(sig_sq, betas)         = sample_NIG(nsim, shape_post, scale_post , mean_post, var_post)
(sig_sq_pri, betas_pri) = sample_NIG(nsim, shape_pri, scale_pri ,mean_pri, var_pri)

# -----------------------------------------------------------------------------
# Independent Normal Inverse Gamma Posterior Sampling
# -----------------------------------------------------------------------------

# Gibbs Sampling settings
ndraws         = 5000
burn           = ndraws - nsim + 1
shape_post_ind = shape_pri + length(Y)
int_prec       = .01

#Fill in
post_beta_gibb = ones(ndraws)
post_prec_gibb = ones(ndraws)

#Draw samples of beta and precision conditional on the previous draw of the other
for i in 1:ndraws
  #update precision and mean parameters
  if i == 1
    prec_gibb = var_pri + int_prec*Y_1'*Y_1
    mean_gibb = inv(prec_gibb)*(var_pri*mean_pri + int_prec*Y_1'*Y)
  else
    prec_gibb = var_pri + post_prec_gibb[i-1,1]*Y_1'*Y_1
    mean_gibb = inv(prec_gibb)*(var_pri*mean_pri + post_prec_gibb[i-1,1]*Y_1'*Y)
  end

  #sample beta
  post_beta_gibb[i,] = rand(Normal(mean_gibb,sqrt(inv(prec_gibb))))

  #update parameters to sample precision matrix
  scale_gibb          = scale_pri + (Y - Y_1*post_beta_gibb[i,1])'*(Y - Y_1*post_beta_gibb[i,1])
  post_prec_gibb[i,1] = rand(Chisq(shape_post_ind))/scale_gibb
end 

#Get rid of the first "burn" samples
beta_gibb    = post_beta_gibb[burn:end,]
prec_gibb    = post_prec_gibb[burn:end,]
sig_sq_gibb  = 1 ./prec_gibb
# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

#Prior and Conjugate Posterior Densities of Beta
p1 = plot(layer(x = betas_pri, Geom.density),
          layer(x = betas, Geom.density, Theme(default_color = "green")),
          Guide.ylabel("Density"),Guide.title("Prior and Posterior Density of β"),
          Coord.cartesian(xmin = -.5, xmax = 1.5, ymin = 0, ymax = 6),
          Guide.manual_color_key("",["Prior","Posterior"],[Gadfly.current_theme().default_color,"green"]))

#Prior and Conjugate Posterior Densities of Sigma Squared
p2 = plot(layer(x = sig_sq_pri, Geom.density),
          layer(x = sig_sq, Geom.density, Theme(default_color = "green")),
          Guide.ylabel("Density"),Guide.title("Prior and Posterior Density of σ²"),
          Coord.cartesian(xmin = 0, xmax = 13, ymin = 0, ymax = 1.3),
          Guide.manual_color_key("",["Prior","Posterior"],[Gadfly.current_theme().default_color,"green"]))

#Contour plot of Normal-Gamma joint prior distribution using samples from conjugate posterior
p3 = plot(layer(z=(x,y) -> 1/(factorial(shape_pri-1)/mean_pri^shape_pri*(2*pi/var_pri)^(1/2))*y^(shape_pri - 1/2)*exp((-1)*y/2*(var_pri*(x - mean_pri)^2 + 2*mean_pri)),
            xmin=[0], xmax=[1], ymin=[2], ymax=[30], Geom.contour),
          layer(x = betas, y = sig_sq, Geom.point),
          Guide.ylabel("σ²"), Guide.xlabel("β"),
          Guide.title("Joint Prior Normal-Gamma Distribution Contour with Sampled Posterior Parameters"))

#Prior and Independent Posterior Densities of Beta
p4 = plot(layer(x = betas_pri, Geom.density),
          layer(x = beta_gibb, Geom.density, Theme(default_color = "darkorchid1")),
          layer(x = betas, Geom.density, Theme(default_color = "green")),
          Guide.ylabel("Density"),
          Guide.title("Prior and Different Posterior Distributions of β"),
          Coord.cartesian(xmin = -.5, xmax = 1.5, ymin = 0, ymax = 6),
          Guide.manual_color_key("",["Prior","Independent Posterior","NIG Posterior"],[Gadfly.current_theme().default_color,"darkorchid1","green"]))

#Conjugate and Independent NIG Posterior Densities
p5 = plot(layer(x = beta_gibb, Geom.density, Theme(default_color = "darkorchid1")),
          layer(x = betas, Geom.density, Theme(default_color = "green")),
          Guide.ylabel("Density"),
          Guide.title("Posterior Distributions with Different Priors of β"),
          Coord.cartesian(xmin = 0, xmax = 0.7, ymin = 0, ymax = 6),
          Guide.manual_color_key("",["Independent NIG","Conjugate NIG"],["darkorchid1","green"]))

#Prior and Both Posterior Densities of Sigma Squared
p6 = plot(layer(x = sig_sq_pri, Geom.density),
          layer(x = sig_sq, Geom.density, Theme(default_color = "green")),
          layer(x = sig_sq_gibb, Geom.density, Theme(default_color = "darkorchid1")),
          Guide.ylabel("Density"),Guide.title("Prior and Posterior Densities of σ²"),
          Coord.cartesian(xmin = 0, xmax = 13, ymin = 0, ymax = 1.7),
          Guide.manual_color_key("",["Prior","Conjugate NIG", "Independent NIG"],[Gadfly.current_theme().default_color,"green","darkorchid1"]))

#Prior and Both Posterior Densities of Sigma Squared
p7 = plot(layer(x = sig_sq, Geom.density, Theme(default_color = "green")),
          layer(x = sig_sq_gibb, Geom.density, Theme(default_color = "darkorchid1")),
          Guide.ylabel("Density"),Guide.title("Conjugate and Independent Posterior Densities of σ²"),
          Coord.cartesian(xmin = 6, xmax = 13, ymin = 0, ymax = 0.5),
          Guide.manual_color_key("",["Conjugate NIG", "Independent NIG"],["green","darkorchid1"]))

#Contour plot of Conjugate NIG prior distribution using samples from conjugate posterior
p8 = plot(layer(z=(x,y) -> 1/(factorial(shape_pri-1)/mean_pri^shape_pri*(2*pi/var_pri)^(1/2))*y^(shape_pri - 1/2)*exp((-1)*y/2*(var_pri*(x - mean_pri)^2 + 2*mean_pri)),
            xmin=[0], xmax=[1], ymin=[2], ymax=[30], Geom.contour),
          layer(x = beta_gibb, y = sig_sq_gibb, Geom.point, Theme(default_color = "darkorchid1")),
          layer(x = betas, y = sig_sq, Geom.point, Theme(default_color = "green")),
          Guide.ylabel("σ²"), Guide.xlabel("β"),
          Guide.title("Joint Prior Independent Normal-Gamma Distribution Contour with Sampled Posterior Parameters"),
          Guide.manual_color_key("Scatter",["Conjugate NIG", "Independent NIG"],["green","darkorchid1"]))

display(p1)
display(p2)
display(p3)
display(p4)
display(p5)
display(p6)
display(p7)
display(p8)
