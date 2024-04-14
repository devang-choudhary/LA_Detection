import repodownloader
import is_funtionality
import is_modify
import set_funtionality
import set_modify
import get_funtionality
import get_modify
import algo5
import algo6
import partition
import json

file = 'projects_list.json'

with open(file, 'r') as f:
    data = json.load(f)
    for project,url in data.items():
        # repodownloader.extract_java_methods_from_project(url, project)
        # partition.partition_methods(project)
        get_funtionality.get_labeled(project)
        get_modify.get_modify(project)
        is_funtionality.is_labeled_data(project)
        is_modify.is_modified(project)
        set_funtionality.set_labeled(project)
        set_modify.set_modify(project)
        # add try and catch statement in all 
        
        # algo5.algo5(project)
        # algo6.algo6(project)
        