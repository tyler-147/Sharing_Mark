#= Simulate data forward from the model specified by Caldara, Scotti, and Zhong
2019. Uses equation (9) as model. =#

#load packages
using Distributions, Statistics, Gadfly

# -----------------------------------------------------------------------------
# Function
# -----------------------------------------------------------------------------

function caldara_simulate(nsim::Integer, c::Float64, β::Float64, b1::Float64, α::Float64,
                          θ::Float64, d1::Float64, ζ::Float64, S::Float64,
                          z0::Float64, h0::Float64)
         #= Simulate data forward using specification from Caldara Scotti, and
         Zhong (2019), using (9) as the model.
         nsim = number of steps simulating forward
         c    = intercept of data simultion
         β    = coefficient of lagged data on data
         b1   = coefficient of lagged stochastic volatility component on data
         α    = intercept of stochastic volatility simulation
         Θ    = coefficient of lagged stochastic volatility component on
                stochastic volatility
         d1   = coefficient of lagged data on stochastic volatility
         S    = multiplier of error on stochastic volatility
         ζ    = correlation between errors
         z0   = initial data
         h0   = initial stochastic volatility component
         Returns a time series of simulated stochastic volatility component (h)
         and simulted data (z). =#

    #Create fill vectors
    h , z      = zeros(nsim), zeros(nsim)
    h[1], z[1] = h0, z0

    #Forward simulate data
    for i in 2:nsim

        #Simulate mean and variance errors
        e , η = rand(MvNormal([0.0; 0.0], [1.0 ζ; ζ 1.0]))

        #Simulate volatility
        h[i] = α + θ*h[i - 1] + d1*z[i - 1] + sqrt(S)*η

        #Simulate data
        z[i] = c + β*z[i - 1] + b1*h[i - 1] + sqrt(exp(h[i]))*e

    end

    return h, z
end

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------

c  = 0.5
β  = 0.5
b1 = 0.05
α  = 0.5
θ  = 0.5
d1 = 0.1
ζ  = 0.1
S  = 0.5
z0 = 0.5
h0 = 0.05

# -----------------------------------------------------------------------------
# Example
# -----------------------------------------------------------------------------

ex = caldara_simulate(100, c, β, b1, α, θ, d1, ζ, S, z0, h0)

plot(layer(y = ex[1], Geom.line),
     layer(y = ex[2], Theme(default_color = "green"), Geom.line),
     Guide.title("Simulated Data with Stochastic Volatility and Correlated Shocks"),
     Guide.manual_color_key("",["Volatility","Data"],["deepskyblue","green"]))
