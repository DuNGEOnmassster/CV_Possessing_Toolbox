import os

image_path = "./test_dataset"


def del_files(path):
    for root , dirs, files in os.walk(path):
        for name in files:
            if name.startswith("._"):
                os.remove(os.path.join(root,name))         
                print ("Delete File: " + os.path.join(root, name))


del_files(image_path)

