from modules import io
import os

if __name__=='__main__':

    config = io.load_yaml('./config/global.yaml')

    cases_dir = config['CASES_DIR']
    cases_prefix = config['DATA_DIR']
    mod_prefs = config['MODALITY']
    type_pref = config['TYPE']
    scaled = config['SCALED']
    cropped = config['CROPPED']

    for mod_pref in mod_prefs:
        cases_dir_save = cases_dir+'_'+mod_pref.lower()#+'_'+type_pref.lower()
        try:
            os.mkdir(cases_dir_save)
        except Exception as e: print(e)

        centerlines = open(cases_dir+'/centerlines.txt')
        centerlines = [f.replace('\n','') for f in centerlines]
        centerlines = ['/centerlines/'+f for f in centerlines]

        surfaces  = open(cases_dir+'/surfaces.txt')
        surfaces  = [f.replace('\n','') for f in surfaces]
        surfaces = ['/surfaces/'+f for f in surfaces]

        modality = open(cases_dir+'/modality.txt')
        modality  = [f.replace('\n','') for f in modality]

        if not cropped:
            if not scaled:
                images = open(cases_dir+'/images.txt').readlines()
                images = [f.replace('\n','') for f in images]
                images = ['/images'+f for f in images]
            else:
                images = open(cases_dir+'/centerlines.txt')
                images = [f.replace('\n','') for f in images]
                images = ['/scaled_images/'+f.replace('vtp','vtk') for f in images]
        
            truths = open(cases_dir+'/truths.txt')
            truths = [f.replace('\n','') for f in truths]
            truths = ['/images'+f for f in truths]
        elif cropped:
            if not scaled:
                images = open(cases_dir+'/centerlines.txt')
                images0 = [f.replace('\n','') for f in images]
                images = ['/cropped_images/cropped/'+f.replace('vtp','vtk') for f in images0]
            else:
                images = open(cases_dir+'/centerlines.txt')
                images0 = [f.replace('\n','') for f in images]
                images = ['/cropped_images/scaled_images/cropped/'+f.replace('vtp','vtk') for f in images0]
            truths = ['/cropped_images/cropped_masks/'+f.replace('vtp','vtk') for f in images0]

        types = open(cases_dir+'/type.txt')
        types  = [f.replace('\n','') for f in types]

        ids = range(len(images))
        if mod_pref != 'ALL':
            ids = [i for i in range(len(modality)) if modality[i] == mod_pref]

        if isinstance(type_pref, str):
            if type_pref != 'ALL':
                ids1 = [i for i in range(len(types)) if types[i] == type_pref]
            else: ids1 = ids
        else:
            ids1 = [i for i in range(len(types)) if types[i] in type_pref]

        ids = [i for i in ids if i in ids1]

        for i in ids:
            d = {}
            d['IMAGE'] = cases_prefix+images[i]
            d['SEGMENTATION'] = cases_prefix+truths[i]
            d['CENTERLINE'] = cases_prefix+centerlines[i]
            d['SURFACE'] = cases_prefix+surfaces[i]

            name = truths[i].split('/')[-1].replace('-cm.mha','')\
                .replace('-image.mha','').replace('_contrast.mha','').replace('.vtk','')

            d['NAME'] = name

            fn = "{}/case.{}.yml".format(cases_dir_save,name)
            #print(d)
            print(fn)
            io.save_yaml(fn,d)

        print('Number of files generated: ', len(ids))
