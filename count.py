import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_items(file_path):
    items = [[] for i in range(11)]
    keys = []
    out_cnt = 0

    with open(file_path, "r") as f:
        for lines in f.readlines():
            line_list = lines.split(", ")
            in_cnt = 0
            for key_value in line_list[1:]:
                key = key_value.split(":")[0]
                value = key_value.split(":")[1]
                print(f"key={key}, value={value}")
                value = float(str(value)[:10])
                items[in_cnt].append(value)

                if out_cnt == 0:
                    keys.append(key)
                    print(f"Add key: {key}")

                in_cnt += 1

            out_cnt += 1

    print(f"keys = {keys}")
    print(f"items: {items}")
    print(f"len items: {len(items[0])}")
    return keys, items


def count_fault(items):
    cnt_fault_img = 0
    cnt_fault_label = 0
    for i in range(len(items[0])):
        if items[0][i] < items[6][i]:
            cnt_fault_img += 1
            print(f"fault img: img: {items[0][i]}, noise img: {items[6][i]}, img < noise is {items[0][i] < items[6][i]}")
        if items[4][i] > items[10][i]:
            cnt_fault_label += 1
            print("fault label")
    print(f"fault img: {cnt_fault_img} / {len(items[0])}, fault label: {cnt_fault_label} / {len(items[0])}")


def draw_box(keys, items):
    # df = {keys[i]:items[i] for i in range(11)}
    # df = pd.DataFrame(df)
    # print(df)
    #设置绘图风格
    plt.style.use('ggplot')
    #处理中文乱码
    # plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    #坐标轴负号的处理
    plt.rcParams['axes.unicode_minus']=False
    plt.figure(figsize=(10,5))#设置画布的尺寸
    plt.title('Examples of boxplot',fontsize=20)#标题，并设定字号大小
    plt.boxplot([items[i] for i in range(11)], labels = keys)#grid=False：代表不显示背景中的网格线
    plt.show()


def draw_violin(keys, items):
    #设置绘图风格
    plt.style.use('ggplot')
    #处理中文乱码
    # plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    #坐标轴负号的处理
    plt.rcParams['axes.unicode_minus']=False

    df = pd.DataFrame({keys[i]:items[i] for i in range(11)})
    # print(df)
    sns.violinplot(x = "类别的one-hot值的信息熵", # 指定x轴的数据
               y = "噪声类别值的信息熵", # 指定y轴的数据
            #    hue = "噪声类别值的信息熵", # 指定分组变量
               data = df, # 指定绘图的数据集
               order = keys, # 指定x轴刻度标签的顺序
               scale = 'count', # 以男女客户数调节小提琴图左右的宽度
               split = True, # 将小提琴图从中间割裂开，形成不同的密度曲线；
               palette = 'RdBu' # 指定不同性别对应的颜色（因为hue参数为设置为性别变量）
              )
    # 添加图形标题
    plt.title('Examples of violin',fontsize=20)
    # 设置图例
    plt.legend(loc = 'upper left', ncol = 2)
    #控制横纵坐标的值域
    plt.axis([-1,4,-10,70])
    # 显示图形
    plt.show()
    


if __name__ == "__main__":
    file_path = "test_dataset/full.txt"
    keys,items = get_items(file_path)
    count_fault(items)
    draw_box(keys,items)
    draw_violin(keys, items)