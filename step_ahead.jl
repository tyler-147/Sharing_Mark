#Step ahead function and showing that it works

#required packages
using Distributions, Gadfly

# -----------------------------------------------------------------------------
# Required functions
# -----------------------------------------------------------------------------
function step_ahead_(beta, variance, Y0, nsim, steps)
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

# -----------------------------------------------------------------------------
# Example
# -----------------------------------------------------------------------------

#Example parameters
beta     = .5
variance = 2
Y0       = 1
nsim     = 10000
steps    = 5

#Apply function
forecast = step_ahead_(beta, variance, Y0, nsim, steps)

#Estimate first and second moments for each step
means     = ones(steps)
variances = ones(steps)

for i in 1:steps
    means[i]     = mean(forecast[i,:])
    variances[i] = var(forecast[i,:])
end

# -----------------------------------------------------------------------------
# Expected Results
# -----------------------------------------------------------------------------

#Calculate expected mean and variance for each step
exp_means        = ones(steps)
exp_means[1]     = Y0
exp_variances    = ones(steps)
exp_variances[1] = 0

for i in 2: steps
    exp_means[i]     = Y0*beta^(i-1)
    exp_variances[i] = variance*(1 - beta^(2*(i - 1)))/(1 - beta^2)
end

#Draw samples from expected distributions
exp_forecast = ones(steps, nsim)

for i in 1:steps
    exp_forecast[i,:] = rand(Normal(exp_means[i], sqrt(exp_variances[i])),nsim)
end

# -----------------------------------------------------------------------------
# Plots to compare distributions of forecasts and simulated expected values
p1 = plot(layer(x = exp_forecast[2,:], Geom.density, Theme(default_color = "green")),
          layer(x = forecast[2,:], Geom.density),
          Guide.ylabel("Density"), Guide.title("Density of First Step Ahead Forecast and Expected Foreast"),
          Guide.manual_color_key("",["First Step Ahead", "Expected"],[Gadfly.current_theme().default_color,"green"]))

p2 = plot(layer(x = exp_forecast[3,:], Geom.density, Theme(default_color = "green")),
          layer(x = forecast[3,:], Geom.density),
          Guide.ylabel("Density"), Guide.title("Density of Second Step Ahead Forecast and Expected Foreast"),
          Guide.manual_color_key("",["Second Step Ahead", "Expected"],[Gadfly.current_theme().default_color,"green"]))

p3 = plot(layer(x = exp_forecast[4,:], Geom.density, Theme(default_color = "green")),
          layer(x = forecast[4,:], Geom.density),
          Guide.ylabel("Density"), Guide.title("Density of Third Step Ahead Forecast and Expected Foreast"),
          Guide.manual_color_key("",["Third Step Ahead", "Expected"],[Gadfly.current_theme().default_color,"green"]))

p4 = plot(layer(x = exp_forecast[5,:], Geom.density, Theme(default_color = "green")),
          layer(x = forecast[5,:], Geom.density),
          Guide.ylabel("Density"), Guide.title("Density of Fourth Step Ahead Forecast and Expected Foreast"),
          Guide.manual_color_key("",["Fourth Step Ahead", "Expected"],[Gadfly.current_theme().default_color,"green"]))

display(p1)
display(p2)
display(p3)
display(p4)
