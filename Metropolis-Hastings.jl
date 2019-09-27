#Metropolis Hastings Algorithm, estimating bivariate normal distribution

# load packages
using Statistics, Distributions, Gadfly, Random, LinearAlgebra

# -----------------------------------------------------------------------------
# Simulation settings
# -----------------------------------------------------------------------------

nsim  = 4000                #number of simulations drawn from algorithm
μ     = [1.0; 2.0][:]       #mean of data
Σ     = [1 .9; .9 1]        #covariance of data
inv_Σ = inv(Σ)              #inverse of covariance of data
δ1    = .75                 #boundary of first uniform coordinate
δ2    = 1.0                 #boundary of second uniform coordinate
δ4    = 1.0                 #boundary for fourth question
var1  = .6                  #variance for first element
var2  = .4                  #variance for second element
c     = .9                  #constant for pseudorejection sampling
D     = [2.0 0.0; 0.0 2.0]  #matrix for pseudorejection sampling

#Set random seed
Random.seed!(8675309)

# -----------------------------------------------------------------------------
# Required function
# -----------------------------------------------------------------------------

function move_prob_Chib(x::Array{Float64,1},y::Array{Float64,1},
                        μ::Array{Float64,1},inv_Σ::Array{Float64,2})
    #= Returns the variable to use in the next simulation, given old and new
       new candidate. If new candidate is accepted, y is returned. If not, x
       is returned.
       x = previous sample
       y = new sample
       μ = mean vector
       Σ = covariance matrix
       Returns x if the new canditiate is rejected or y if the new candidate
       is accepted.=#

       #estimate probability of a move to new candidate
   prob = exp(-1 / 2 * (y - μ)' * inv_Σ * (y - μ)) /
          exp(-1 / 2 * (x - μ)' * inv_Σ * (x - μ))
   α    = min(prob, 1)

       #generate random sample from a uniform distribution [0,1]
   u = rand(Uniform(0, 1))

       #compare u and α
   if u <= α

      return y

   else

      return x

   end
end

function random_walk_MH(nsim::Integer, μ::Array{Float64,1},
                        inv_Σ::Array{Float64,2}, δ1::Float64,
                        δ2::Float64, option::Integer)
   #= Estimates Metropolis-Hastings for bivariate normal distribution pulling new
      candidates with a random walk processed.
      nsim   = number of desired accepted simulations
      μ      = mean of data
      inv_Σ  = the inverse of the covariance of data
      δ1     = boundary on uniform distribution/ variance of first element
      δ2     = boundary on uniform distribution/ variance of second element
      δ3     = boundary on uniform distribution
      option = if 1, uses uniform distribution for random error for new
               candidate. if 2, uses bivariate normal distribution with no
               covariance between two elements
      Returns an (2, nsim) array of simulated data. =#
   #set up fill in vectors
   x = zeros(2, nsim)

   #fill in with guesses
   for i in 1:nsim-1
      #Estiamte error in random walk
      if option == 1

         error = [rand(Uniform(-δ1, δ1)); rand(Uniform(-δ2, δ2))]

      elseif option == 2

         error = [rand(Normal(0, sqrt(δ1))); rand(Normal(0, sqrt(δ2)))]

      elseif option == 4

         error = [rand(Uniform(-δ4, δ4)); rand(Uniform(-δ4, δ4))]

      end

      #Estimate new candidate
      if (option == 1) | (option == 2)

         y = x[:, i] + error

      elseif option == 4

         y = μ - (x[:, i] - μ) + error

      end

      #Test new candidate
      cand = move_prob_Chib(x[:, i], y, μ, inv_Σ)

      #Insert new candidate if it passes
      if cand == y

         x[:, i+1] = y

      else

         x[:, i+1] = x[:,i]

      end

   end

   return x
end

# -----------------------------------------------------------------------------
# Example 1.1
# -----------------------------------------------------------------------------

x1 = random_walk_MH(nsim, μ, inv_Σ, δ1, δ2, 1)

# -----------------------------------------------------------------------------
# Example 1.2
# -----------------------------------------------------------------------------

x2 = random_walk_MH(nsim, μ, inv_Σ, var1, var2, 2)

# -----------------------------------------------------------------------------
# Example 1.3
# -----------------------------------------------------------------------------

#Array of samples to fill in
x3 = zeros(2, nsim)

#Adjusted matrix
D_adj = norm(D)^(-1/2)

#Tracking variables

for i in 1:nsim - 1
   #Generate random uniform u
   u = rand(Uniform(0,1))

   #Generate new candidate from bivariate independent normal distribution
   a = 0

   while a<1

      #Generate candidate
      z = rand(MvNormal(zeros(2), [1.0 0.0; 0.0 1.0]))

      #Dominating function and proportional function
      ch_z = c/(2*pi)*D_adj*exp(-1/2*(z - μ)'*D*(z - μ))
      f_z  = exp(-1/2*(z - μ)'*inv_Σ*(z - μ))
      r    = f_z/ch_z

      if r > u

         global y = z

         a = 1

      end
   end

   #Dominating function
   ch_x = c/(2*pi)*D_adj*exp(-1/2*(x3[:,i] - μ)'*D*(x3[:,i] - μ))
   ch_y = c/(2*pi)*D_adj*exp(-1/2*(y - μ)'*D*(y - μ))

   #Proportional function
   f_x = exp(-1/2*(x3[:,i] - μ)'*inv_Σ*(x3[:,i] - μ))
   f_y = exp(-1/2*(y - μ)'*inv_Σ*(y - μ))

   #compare proportional and dominating functions
   if f_x < ch_x

      α = 1

   elseif (f_x >= ch_x) & (f_y < ch_y)

      α = ch_x/f_x

   elseif (f_x >= ch_x) & (f_y >= ch_y)

      α = min(f_y*ch_x/(f_x*ch_y), 1)

   end

   #Compare u and α
   if u <=  α

      x3[:, i + 1] = y

   else

      x3[:, i+1] = x3[:,i]

   end
end

# -----------------------------------------------------------------------------
# Example 1.4
# -----------------------------------------------------------------------------

x4 = random_walk_MH(nsim, μ, inv_Σ, δ4, δ4, 4)
