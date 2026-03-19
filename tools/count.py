import os

folder = r"C:\Users\15972\Desktop\icra2026\results\prompt"
threshold = 50 * 1024  # 50KB (单位: 字节)

count = 0
big_files = []

for fname in os.listdir(folder):
    fpath = os.path.join(folder, fname)
    if os.path.isfile(fpath) and fname.lower().endswith(".txt"):
        size = os.path.getsize(fpath)  # 字节
        if size > threshold:
            count += 1
            big_files.append((fname, size))

print(f"超过 50KB 的 txt 文件数量: {count}")
print("前几个大文件示例:")
for name, size in big_files[:10]:  # 只打印前10个
    print(f"  {name} - {size/1024:.1f} KB")
