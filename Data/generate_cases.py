from modules import io


config = io.load_yaml('./config/global.yaml')

cases_dir    = config['CASES_DIR']
cases_prefix = config['CASES_PREFIX']

images = open(cases_dir+'/images.txt').readlines()
images = [f.replace('\n','') for f in images]

truths = open(cases_dir+'/truths.txt')
truths = [f.replace('\n','') for f in truths]

groups = open(cases_dir+'/groups.txt')
groups = [f.replace('\n','') for f in groups]

paths  = open(cases_dir+'/paths.txt')
paths  = [f.replace('\n','') for f in paths]

for i in range(len(images)):
    d = {}
    d['IMAGE'] = cases_prefix+images[i]
    d['SEGMENTATION'] = cases_prefix+truths[i]
    d['GROUPS'] = cases_prefix+groups[i]
    d['PATHS'] = cases_prefix+paths[i]

    name = images[i].split('/')[-1].replace('-cm.mha','')\
        .replace('-image.mha','').replace('_contrast.mha','')

    d['NAME'] = name

    fn = "{}/case.{}.yml".format(cases_dir,name)
    print(d)
    print(fn)
    io.save_yaml(fn,d)
