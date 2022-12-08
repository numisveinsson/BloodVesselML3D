from modules import io
import os

config = io.load_yaml('./config/global.yaml')

cases_dir_save = config['CASES_DIR']+'_'+config['MODALITY']
cases_dir = config['CASES_DIR']
try:
    os.mkdir(cases_dir_save)
except Exception as e: print(e)

cases_prefix = config['DATA_DIR']
mod_pref = config['MODALITY']

images = open(cases_dir+'/images.txt').readlines()
images = [f.replace('\n','') for f in images]
images = ['/images'+f for f in images]

truths = open(cases_dir+'/truths.txt')
truths = [f.replace('\n','') for f in truths]
truths = ['/images'+f for f in truths]

centerlines = open(cases_dir+'/centerlines.txt')
centerlines = [f.replace('\n','') for f in centerlines]
centerlines = ['/centerlines/'+f for f in centerlines]

surfaces  = open(cases_dir+'/surfaces.txt')
surfaces  = [f.replace('\n','') for f in surfaces]
surfaces = ['/surfaces/'+f for f in surfaces]

modality = open(cases_dir+'/modality.txt')
modality  = [f.replace('\n','') for f in modality]

ids = range(len(images))
if mod_pref != 'ALL':
    ids = [i for i in range(len(modality)) if modality[i] == mod_pref]

for i in ids:
    d = {}
    d['IMAGE'] = cases_prefix+images[i]
    d['SEGMENTATION'] = cases_prefix+truths[i]
    d['CENTERLINE'] = cases_prefix+centerlines[i]
    d['SURFACE'] = cases_prefix+surfaces[i]

    name = truths[i].split('/')[-1].replace('-cm.mha','')\
        .replace('-image.mha','').replace('_contrast.mha','')

    d['NAME'] = name

    fn = "{}/case.{}.yml".format(cases_dir_save,name)
    print(d)
    print(fn)
    io.save_yaml(fn,d)

print('Number of files generated: ', len(ids))
