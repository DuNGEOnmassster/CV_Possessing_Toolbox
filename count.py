import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_items(file_path):
    with open(file_path, "r") as f:
        for lines in f.readlines():
            line_list = lines.split(", ")
            for key_value in line_list[1:]:
                key = key_value.split(":")[0]
                value = key_value.split(":")[1]
                print(f"key={key}, value={value}")


if __name__ == "__main__":
    file_path = "test_dataset/test.txt"
    get_items(file_path)