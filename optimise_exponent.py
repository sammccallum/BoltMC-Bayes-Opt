from boltmc.simulation import Simulation
from boltmc.recorders import RecordMeanVelocity
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import numpy as np
import ruamel.yaml
import json

def get_midpoint(path):
    yaml = ruamel.yaml.YAML()
    yaml.preserve_quotes = True
    with open(path) as fp:
        data = yaml.load(fp)

    midpoint = np.array([data['el_const'], data['eps_inf'], data['eps_s'], data['hwo'][0], data['mstar'][0][0], data['acodefpot'][0]])

    return midpoint

def uniform_prior(midpoint):
    min_percentage = 0.8
    max_percentage = 1.2
    mask = np.random.uniform(min_percentage, max_percentage, len(midpoint))
    sample = midpoint * mask
    return sample

def edit_params(path, params):
    yaml = ruamel.yaml.YAML()
    yaml.preserve_quotes = True
    with open(path) as fp:
        data = yaml.load(fp)
    
    data['el_const'] = float('%.4g' % params[0])
    data['eps_inf'] = float('%.4g' % params[1])
    data['eps_s'] = float('%.4g' % params[2])
    data['hwo'][0] = float('%.4g' % params[3])
    data['mstar'][0][0] = float('%.4g' % params[4])
    data['acodefpot'][0] = float('%.4g' % params[5])

    # write data to .yaml file
    with open(path, "w") as fp:
        yaml.dump(data, fp)

def edit_key_yaml(path, key, value):
    yaml = ruamel.yaml.YAML()
    yaml.preserve_quotes = True
    with open(path) as fp:
        data = yaml.load(fp)
    
    data[key] = value
    # write data to .yaml file
    with open(path, "w") as fp:
        yaml.dump(data, fp)

def run_simulation(infile_path, outfile_path):
    s = Simulation(infile=infile_path)

    s.add_extensions(
    [
        RecordMeanVelocity(s, outfile=outfile_path),
    ]
    )

    s.run()

def write_params_to_csv(path, params):
    params = params.reshape(1,-1)
    with open(path,'a') as csv:
        np.savetxt(csv, params, delimiter=',', fmt='%.4g')

def calculate_mobility(path):
    with open(path, 'r') as f:
        data = json.load(f)

    data_length = len(data["mean_velocity"])
    x_vel = np.zeros((1, data_length))

    for i in range(0, data_length):
        x_vel[0,i] = data["mean_velocity"][i][0]   

    x_vel = x_vel[:,10:]
    mean_vel = np.mean(x_vel)
    mean_vel = abs(mean_vel)
    mobility = mean_vel / (1.0e+5)
    return mobility

def calculate_exponent_dataset(path):
    dataset = np.genfromtxt(path, delimiter=',')
    dataset_length = dataset.shape[0]
    temps = [200, 250, 300]
    coefs = np.zeros((dataset_length, 1))
    
    for i in range(dataset_length):
        coefs[i] = calculate_exponent(temps, dataset[i])

    dataset_exp = np.concatenate((dataset, coefs), axis=1)
    np.savetxt('dataset_exp.csv', dataset_exp, delimiter=',', fmt='%.4g')

def calculate_exponent(temps, params):
    temps = np.log10(temps).reshape(-1, 1)
    mobs = np.log10(params[6:9]).reshape(-1, 1)
    reg = LinearRegression().fit(temps, mobs)
    return reg.coef_[0]

def fit_gpr(path):
    dataset = np.genfromtxt(path, delimiter=',')
    X = dataset[:, :6]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = dataset[:, 9]

    kernel = 1 * RBF(np.ones(6)) + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X, y)

    return gpr, scaler

def random_params_set(midpoint):
    n_set = 50000
    params_set = np.zeros((n_set, len(midpoint)))

    for i in range(n_set):
        params_set[i] = uniform_prior(midpoint)

    return params_set

def acquisition_function(mean, std, min_exp):
    alpha = 0.05
    normal = norm(mean, std)
    prob = normal.cdf(min_exp - alpha)
    return prob

def bayes_opt(path, midpoint):
    while True:
        gpr, scaler = fit_gpr(path)
        params_set = random_params_set(midpoint)
        params_set = scaler.transform(params_set)
        gpr_mean, gpr_std = gpr.predict(params_set, return_std=True)

        min_exp = get_min_exp(path)
        prob = acquisition_function(gpr_mean, gpr_std, min_exp)
        idx = np.argmax(prob)
        
        params_set = scaler.inverse_transform(params_set)
        opt_params = params_set[idx]

        run_three_temps(opt_params)


def run_three_temps(params):
    infile_path = 'infile_polaron.yaml'
    temps = [200, 250, 300]
    edit_params('mapbi3_vary.yaml', params)

    for temp in temps:
        edit_key_yaml(infile_path, 'lattice_temperature', temp)
        run_simulation(infile_path, 'meanvel_temp.json')
        mobility = calculate_mobility('meanvel_temp.json')
        params = np.append(params, mobility)
            
    exponent = calculate_exponent(temps, params)
    params = np.append(params, exponent)
        
    write_params_to_csv('dataset_exp.csv', params)

def get_min_exp(path):
    dataset = np.genfromtxt(path, delimiter=',')
    y = dataset[:, 9]
    return np.min(y)

if __name__ == "__main__":
    midpoint = get_midpoint('mapbi3.yaml')
    dataset_path = 'dataset_exp.csv'
    bayes_opt(dataset_path, midpoint)
