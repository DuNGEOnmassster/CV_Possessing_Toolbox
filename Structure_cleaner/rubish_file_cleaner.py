'''
Rewrite the path, adjust the prefixes and suffixes as you wish
'''

import os

image_path = "./"
clean_by_prefixes = True
clean_by_suffixes = False

prefixes = ".DS_S"
suffixes = ".png"

def del_files(path):
    for root , dirs, files in os.walk(path):
        for name in files:
            if clean_by_prefixes:
                if name.startswith(prefixes):
                    os.remove(os.path.join(root,name))         
                    print ("Delete File: " + os.path.join(root, name))

            if clean_by_suffixes:
                if name.endswith(suffixes):
                    os.remove(os.path.join(root,name))         
                    print ("Delete File: " + os.path.join(root, name))


del_files(image_path)