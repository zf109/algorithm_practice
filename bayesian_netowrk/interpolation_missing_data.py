import pandas as pd

# we use 1 and 0 to represent True and False for reasons that will become clear later
full = pd.DataFrame({
    'swims_like_a_duck':  [0,0,0,0,1,1,1,1, 1, 1],
    'quacks_like_a_duck': [0,1,0,1,0,1,0,1, 0, 1],
    'duck':               [0,0,0,0,0,1,0,1, 0, 1]
})

with_missing = pd.DataFrame({
    'swims_like_a_duck':  [0,0,0,0,1,1,1,1,  1, -1],
    'quacks_like_a_duck': [0,1,0,1,0,1,0,1, -1, -1],
    'duck':               [0,0,0,0,0,1,0,1,  0,  1]
})


import pymc

print("------building network------")

# prior probabilities for swimming and quacking
swim_prior = pymc.Uniform('P(swims)', lower=0, upper=1, size=1)
quack_prior = pymc.Uniform('P(quacks)', lower=0, upper=1, size=1)

# probability of being a duck conditional on swimming and quacking
# (or not swimming and quacking etc.)
p_duck_swim_quack = pymc.Uniform('P(duck | swims & quacks)', lower=0, upper=1, size=1)
p_duck_not_swim_not_quack = pymc.Uniform('P(duck | not swims & not quacks)', lower=0, upper=1, size=1)
p_duck_not_swim_quack = pymc.Uniform('P(duck | not swims & quacks)', lower=0, upper=1, size=1)
p_duck_swim_not_quack = pymc.Uniform('P(duck | swims & not quacks)', lower=0, upper=1, size=1)

import numpy as np
swim_data = with_missing.swims_like_a_duck
masked_swim_data = np.ma.masked_array(swim_data, swim_data == -1, fill_value=0)

quack_data = with_missing.quacks_like_a_duck
masked_quack_data = np.ma.masked_array(quack_data, quack_data == -1, fill_value=0)

duck_data = with_missing.duck
masked_duck_data = np.ma.masked_array(duck_data, duck_data == -1, fill_value=0)

print("masked_quack_data:", masked_quack_data)


# number of animal observations
n = len(with_missing)

# with 'size=n' with tell pymc that 'swims' is actually a sequence of n Bernoulli variables
swims = pymc.Bernoulli('swims', p=swim_prior, observed=True, value=masked_swim_data, size=n)
quacks = pymc.Bernoulli('quacks', p=quack_prior, observed=True, value=masked_quack_data, size=n)


# auxiliary pymc variable - probability of duck
@pymc.deterministic
def duck_probability(
        swims=swims,
        quacks=quacks,
        p_duck_swim_quack=p_duck_swim_quack,
        p_duck_not_swim_quack=p_duck_not_swim_quack,
        p_duck_swim_not_quack=p_duck_swim_not_quack,
        p_duck_not_swim_not_quack=p_duck_not_swim_not_quack):

    d = []
    for s, q in zip(swims, quacks):
        if (s and q):
            d.append(p_duck_swim_quack)
        elif (s and (not q)):
            d.append(p_duck_swim_not_quack)
        elif ((not s) and q):
            d.append(p_duck_not_swim_quack)
        elif ((not s) and (not q)):
            d.append(p_duck_not_swim_not_quack)
        else:
            raise ValueError('this should never happen')

    return np.array(d).ravel()

# AND FINALLY
duck = pymc.Bernoulli('duck', p=duck_probability, observed=True, value=masked_duck_data, size=n)

# putting it all together
model = pymc.Model([swims, quacks, duck])


print("------before fit------")
print("swims.value:", swims.value)
print("quacks.value:", quacks.value)
print("duck.value:", duck.value)
print("swim_prior.value:", swim_prior.value)
print("quack_prior.value:", quack_prior.value)
print("p_duck_not_swim_quack.value:", p_duck_not_swim_quack.value)
print("p_duck_swim_quack.value:", p_duck_swim_quack.value)

pymc.MAP(model).fit()

print("------after fit------")
print("swims.value:", swims.value)
print("quacks.value:", quacks.value)
print("duck.value:", duck.value)
print("swim_prior.value:", swim_prior.value)
print("quack_prior.value:", quack_prior.value)
print("p_duck_not_swim_quack.value:", p_duck_not_swim_quack.value)
print("p_duck_swim_quack.value:", p_duck_swim_quack.value)