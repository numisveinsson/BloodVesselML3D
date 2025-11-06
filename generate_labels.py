import os
import glob
import pandas

from modules import io
from modules.pre_process import *

if __name__=='__main__':

    keep_values = ['NAME','RADIUS', 'NUM_OUTLETS', 'BIFURCATION']

    fns = ['_train', '_val']

    global_config_file = "./config/global.yaml"
    global_config = io.load_yaml(global_config_file)
    modality = global_config['MODALITY'].lower()

    data_folder = global_config['OUT_DIR']

    output_folder  = data_folder + 'labels/'
    try:
        os.mkdir(output_folder)
    except Exception as e: print(e)

    # data_folder = '/Users/numisveins/Library/Mobile Documents/com~apple~CloudDocs/Documents/Side_SV_projects/SV_ML_Training/3d_ml_data/test4/'
    # modality = 'ct'

    csv_file = "_Sample_stats.csv"
    csv_list = pandas.read_csv(data_folder+modality+csv_file)

    for fn in fns:
        imgVol_name = []
        for subject_dir in natural_sort(glob.glob(os.path.join(data_folder,modality+fn,'*.nii.gz')) \
                +glob.glob(os.path.join(data_folder,modality+fn,'*.nii')) ):
            imgVol_name.append(os.path.realpath(subject_dir).replace(os.path.join(data_folder,modality+fn)+'/', '').replace('_0.nii.gz',''))
        print("number of data points %d" % len(imgVol_name))

        csv_array = csv_list.loc[csv_list['NAME'].isin(imgVol_name)]
        csv_list_small = csv_array[keep_values]

        csv_list_small.to_csv(output_folder+modality+fn+'_labels.csv')


    import pdb; pdb.set_trace()
    # Second csv only for size and direction
    keep_values = ['RADIUS', 'TANGENTX', 'TANGENTY', 'TANGENTZ', 'BIFURCATION']
    csv_list_small = []
    for data in csv_list:
        data1 = {the_key: data[the_key] for the_key in keep_values}
        csv_list_small.append(data1)
    csv_file = "_Regression_labels.csv"
    with open(out_dir+modality+csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keep_values)
        writer.writeheader()
        for data in csv_list_small:
            writer.writerow(data)
