import yaml
import numpy as np

# A yaml constructor is for loading from a yaml node.
# This is taken from @misha 's answer: http://stackoverflow.com/a/15942429
def opencv_matrix_constructor(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    mat = np.array(mapping["data"])
    mat.resize(mapping["rows"], mapping["cols"])
    return mat

# A yaml representer is for dumping structs into a yaml node.
# So for an opencv_matrix type (to be compatible with c++'s FileStorage) we save the rows, cols, type and flattened-data
def opencv_matrix_representer(dumper, mat):
    if len(mat.shape)>1: cols=int(mat.shape[1])
    else: cols=1
    mapping = {'rows': int(mat.shape[0]), 'cols': cols, 'dt': 'd', 'data': mat.reshape(-1).tolist()}
    return dumper.represent_mapping(u"tag:yaml.org,2002:opencv-matrix", mapping)

def load_yaml_data(data_name):
    yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", opencv_matrix_constructor)
    yaml.add_representer(np.ndarray, opencv_matrix_representer) 
    with open(data_name, "r") as file:
        data = yaml.load(file)
    return data

        