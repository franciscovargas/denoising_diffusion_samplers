from os import path


root_path = path.dirname(path.abspath(path.dirname(__file__)))

data_path = path.join(root_path, "data")
pines_path = path.join(data_path, "fpines.csv")
lr_sonar_path = path.join(data_path, "ionosphere_full.pkl")
ion_path =  path.join(data_path, "sonar_full.pkl")
vae_path= path.join(data_path, "vae.pickle")

results_path = data_path = path.join(data_path, "results")