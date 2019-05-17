"""
Practice on calculating/answering some stats questions
"""

# prob = dbinom(x=0, size = 1:2400, prob = 1/2000)
n_sample = 200

n = 2000
x = 10
k = 10
prod(n-k-0:(x-1))/prod(n-0:(x-1))

get_prob <- function(x, n, k){
  prod(n-k-0:(x-1))/prod(n-0:(x-1))
}
prob = get_prob(x, n , k)

prob = choose(n-x, k)/choose(n, k)
plot(prob)

# choose(n-x, k)/choose(n, k)
