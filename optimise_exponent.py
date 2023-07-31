from boltmc.simulation import Simulation
from boltmc.recorders import RecordMeanVelocity
import numpy as np
import ruamel.yaml
import time
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
    # stderror_vel = np.std(x_vel) / ((1.0e+5) * np.sqrt(x_vel.shape[1]))
    # std_vel = np.std(x_vel) / ((1.0e+5) * np.sqrt(x_vel.shape[1]))
    mean_vel = np.mean(x_vel)
    mean_vel = abs(mean_vel)
    mobility = mean_vel / (1.0e+5)
    return mobility

if __name__ == "__main__":
    infile_path = 'infile_polaron.yaml'
    midpoint = get_midpoint('mapbi3.yaml')
    n_runs = 100
    temps = [200, 250, 300]

    for n in range(n_runs):
        params = uniform_prior(midpoint)
        edit_params('mapbi3_vary.yaml', params)

        for temp in temps:
            edit_key_yaml(infile_path, 'lattice_temperature', temp)
            run_simulation(infile_path, 'meanvel_temp.json')
            mobility = calculate_mobility('meanvel_temp.json')
            params = np.append(params, mobility)
        
        write_params_to_csv('dataset.csv', params)
