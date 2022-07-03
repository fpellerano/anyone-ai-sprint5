
#import yaml
#with open('experiments/exp_001/config.yml', 'r') as stream:
#    try:
#        #print(yaml.safe_load(stream))
#        data_aug_l = yaml.safe_load(stream)
#        #print(data_aug_layer)
#        print(data_aug_l['data']['directory'])
#        #print(**data_aug_l['epochs'])
#    except yaml.YAMLError as exc:
#        print(exc)
import os

path ='data\car_ims_v1'
alloflines = [] 

for root,subdirs,files in os.walk(path):
    for file in files:
        if file.split('.')[-1]=='jpg':
            print(root)