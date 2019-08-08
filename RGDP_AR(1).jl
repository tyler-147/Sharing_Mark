#AR(1) bayesian regression of RGDP growth rates with plots

#load packages
import CSV, DataFrames, Distributions, Statistics
using Gadfly

#load RGDP growth FRED data from csv
if homedir() == "C:\\Users\\tgwin"

  location_data = "C:\\Users\\tgwin\\OneDrive\\Fed\\Mark Bognanni"

elseif homedir() == "C:\\Users\\Mark"

  location_data = "C:\\Users\\Mark\\Documents\\git_repos\\Sharing_Mark"

end

cd(location_data)

frame = CSV.read("RGDP_Practice.csv")
frame_arr = convert(Matrix, frame)
RGDP = convert(Array{Float64}, frame_arr[:,2])
demean_RGDP = RGDP .- Statistics.mean(RGDP)

#Set up lagged data
Y = demean_RGDP[2:end]
Y_1 = demean_RGDP[1:end-1]

#Set up hyperparameters
beta_ = 0.5
h_ = 10
v_ = 7
s_ = 5

#Estimate OLS betas and get SSE
b = inv(Y_1'*Y_1)*Y_1'*Y
errors_sq = (Y - Y_1*b)'*(Y - Y_1*b)

#Update parameters
h = h_ + Y_1'*Y_1
beta = inv(h)*(h_*beta_ + Y_1'*Y_1*b)
v = v_ + length(Y)/2
s = s_ + 1/2*(Y'*Y + beta_*h_*beta_ - beta*h*beta)

#Take n samples of sigma^2 and then sample betas conditional on sigma^2
n = 500
betas = ones(n)
sig_sq = rand(Distributions.InverseGamma(v,s),n)
for i in 1:n
    betas[i,1] = rand(Distributions.Normal(beta,inv(h)*sig_sq[i,1]))
end

#Draw samples from the prior distributions
sig_sq_ = rand(Distributions.InverseGamma(v_,s_),n)
betas_ = ones(n)
for i in 1:n
    betas_[i,1] = rand(Distributions.Normal(beta_,inv(h_)*sig_sq_[i,1]))
end

# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

#Prior and Posterior Densities of Beta
p1 = plot(layer(x = betas_, Geom.density),
          layer(x = betas, Geom.density, Theme(default_color = "green")),
          Guide.ylabel("Density"),Guide.title("Prior and Posterior Density of β"),
          Coord.cartesian(xmin = .2, xmax = .7, ymin = 0, ymax = 80),
          Guide.manual_color_key("",["Prior","Posterior"],[Gadfly.current_theme().default_color,"green"]))

#Prior and Posterior Densities of Sigma Squared
p2 = plot(layer(x = sig_sq_, Geom.density),
          layer(x = sig_sq, Geom.density, Theme(default_color = "green")),
          Guide.ylabel("Density"),Guide.title("Prior and Posterior Density of σ²"),
          Coord.cartesian(xmin = 0, xmax = 13, ymin = 0, ymax = 1.5),
          Guide.manual_color_key("",["Prior","Posterior"],[Gadfly.current_theme().default_color,"green"]))

#Contour plot of Normal-Gamma joint prior distribution and
p3 = plot(layer(z=(x,y) -> 1/(factorial(v_-1)/beta_^v_*(2*pi/h_)^(1/2))*y^(v_ - 1/2)*exp((-1)*y/2*(h_*(x - beta_)^2 + 2*beta_)),
            xmin=[0], xmax=[1], ymin=[2], ymax=[30], Geom.contour),
          layer(x = betas, y = sig_sq, Geom.point),
          Guide.ylabel("σ²"), Guide.xlabel("β"),
          Guide.title("Joint Prior Normal-Gamma Distribution Contour with Sampled Posterior Parameters"))

display(p1)
display(p2)
display(p3)