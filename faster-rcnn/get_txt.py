import os

# 在这里直接修改文件夹路径
folder = r"C:\Users\Administrator\Desktop\dataset_final\images\test"

with open('test.txt', 'w', encoding='utf-8') as f:
    for file in os.listdir(folder):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            f.write(os.path.splitext(file)[0] + '\n')

print("完成！")