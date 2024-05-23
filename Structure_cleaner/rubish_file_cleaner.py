'''
Rewrite the path, adjust the prefixes and suffixes as you wish
'''

import os

image_path = "./"
clean_by_prefixes = True
clean_by_suffixes = False

prefixes = ".DS_S"
suffixes = ".png"

dir_path = "../../output"
dir_names = "point_cloud"

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

def del_dirs(path):
    for root , dirs, files in os.walk(path):
        for name in dirs:
            if dir_names in name:
                try:
                    os.rmdir(os.path.join(root,name))
                    print ("Delete Dir with os.rmdir: " + os.path.join(root, name))
                except OSError:
                    os.system("rm -rf " + os.path.join(root,name))
                    print ("Delete Dir with os.system rm -rf: " + os.path.join(root, name))
                


if __name__ == '__main__':
    # del_files(image_path)
    del_dirs(dir_path)

    # del_files(image_path)