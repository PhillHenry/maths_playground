import pymc3 as pm

basic_model = pm.Model()

with basic_model:

    # Priors for unknown model parameters
    alpha = pm.Beta('mybeta', 65, 42)

map_estimate = pm.find_MAP(model=basic_model)

print(map_estimate) # {'mybeta_logodds__': array(0.44531087), 'mybeta': array(0.60952377)}