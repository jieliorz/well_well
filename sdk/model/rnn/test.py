import yaml

with open('model_params.yml','r') as f:
	data=yaml.load(f)

print(data)