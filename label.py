import os

path = "class_角度"  # 图片集路径
classes = [i for i in os.listdir(path)]
files = os.listdir(path)
train = open("train_angle.txt", 'w')
val = open("val_angle.txt", 'w')
for i in classes:
    s = 0
    for imgname in os.listdir(os.path.join(path, i)):

        if s % 5 != 0:  # 5：1划分训练集测试集
            name = os.path.join(path, i) + '\\' + imgname + ' ' + str(classes.index(i)) + '\n'  # 我是win10,是\\,ubuntu注意！
            train.write(name)
        else:
            name = os.path.join(path, i) + '\\' + imgname + ' ' + str(classes.index(i)) + '\n'
            val.write(name)
        s += 1

val.close()
train.close()