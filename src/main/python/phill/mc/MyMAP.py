import pymc3 as pm

basic_model = pm.Model()

with basic_model:

    # Priors for unknown model parameters
    posterior = pm.Beta('mybeta', 65, 42)

    prior = pm.Beta('prior', 2, 5)
    # obs = pm.Beta('obs', 63, 37)
    # posterior = prior * obs

    product = pm.Potential("potential", posterior * prior)
    product = pm.Deterministic("deterministic", posterior * prior)

map_estimate = pm.find_MAP(model=basic_model)

# {'prior': array(0.22100532), 'mybeta': array(0.61002453), 'prior_logodds__': array(-1.25981743), 'mybeta_logodds__': array(0.44741533)}
# {'mybeta_logodds__': array(0.44741533), 'prior': array(0.22100532), 'prior_logodds__': array(-1.25981743), 'mybeta': array(0.61002453)}
print(map_estimate) # {'mybeta_logodds__': array(0.44531087), 'mybeta': array(0.60952377)}
print(map_estimate['mybeta'])
print(map_estimate['deterministic'])  # 0.13481867
# print(map_estimate['potential'])  # KeyError: 'potential'
print("Finished")
