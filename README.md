# Supporting data and code for "Bayesian optimisation approach to quantify the effect of input parameter uncertainty on predictions of numerical physics simulations" submitted to Applied Machine Learning

- dataset_exp.csv contains, in each row, the six material parameters input to BoltMC, polaron mobilities at 200K, 250K, 300K, and the temperature exponent.
  Each row of data is the result of performing Bayesian Optimisation within a +/- 20% uncertainty range.
- dataset_exp_40.csv contains, with identical formatting to 'dataset_exp.csv', the result of the Bayesian Optimisation method after the uncertainty range has been extended to +/- 40%.
  Note that the data saved in 'dataset_exp.csv' was available to the method during optimisation within the +/- 40% range.
- optimise_exponent.py is a python script for performing the minimisation of the temperature exponent.

An installation of the code BoltMC, a Monte Carlo solver for the Boltzmann transport equation, is also required (see https://gitlab.com/ABW_bath_group/boltmc-eocoe-demo/-/tree/main/).
