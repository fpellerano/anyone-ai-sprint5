import pandas as pd
import os
from tqdm import tqdm

car_dataset_labels = pd.read_csv("data/car_dataset_labels.csv")

c_d_l_columns_names = car_dataset_labels.columns.values
c_d_l_subset = car_dataset_labels['subset'].unique()
c_d_l_class = car_dataset_labels['class'].unique()

# total=car_dataset_labels.shape[0]

for i in c_d_l_subset:
    # print(f"data/{i}")
    newpath = f"data/car_ims_v1/{i}" 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
for _, line in car_dataset_labels.iterrows():
    # if line['img_name']=='016183.jpg':
    if line['subset']=='test':
        name_class_test=line['class'].replace(' ', '_')
        newpath2 = f"data/car_ims_v1/test/{name_class_test}" 
        # print(newpath2)
        os.makedirs(newpath2, exist_ok=True)
        src_test = f'data/car_ims/{line["img_name"]}'
        dst_test = f'data/car_ims_v1/test/{name_class_test}/{line["img_name"]}'
        try:
            os.link(src_test, dst_test)
        except:
                print(f'The image {dst_test} exist in the path.') 
    else:
        if line['subset']=='train':
            name_class_train=line['class'].replace(' ', '_')
            newpath3 = f"data/car_ims_v1/train/{name_class_train}" 
            os.makedirs(newpath3, exist_ok=True)
            src_train = f'data/car_ims/{line["img_name"]}'
            dst_train = f'data/car_ims_v1/train/{name_class_train}/{line["img_name"]}'
            try:
                os.link(src_train, dst_train)
            except:
                print(f'The image {dst_train} exist in the path.') 


~

/mnt/c/users/u189197/desktop/tambo/anyoneai/sprint5/sprint_05_project/sprint5-project/data/car_ims
/home/app/src/data/car_ims/
/mnt/c/users/u189197/desktop/tambo/anyoneai/sprint5/sprint_05_project/sprint5-project/data/car_dataset_labels.csv
/home/app/src/src/data/



/mnt/c/users/u189197/desktop/tambo/anyoneai/sprint5/sprint_05_project/sprint5-project/data/car_ims_v1
/home/app/src/src/data/car_ims_v1/

/home/app/src/scripts/prepare_train_test_dataset.py

final:
python scripts/prepare_train_test_dataset.py /home/app/src/data/car_ims /home/app/src/src/data /home/app/src/src/data/car_ims_v1
python3 scripts/prepare_train_test_dataset.py /home/fpellerano/sprint5-project/data/car_ims /home/fpellerano/sprint5-project/fpellerano/sprint5-project/data /home/fpellerano/sprint5-project/data/car_ims_v1