from pathlib import Path

image_parents = Path("../扫描文件/常用中药饮片识别与应用图谱")
label_parents = Path(r"C:\Users\Administrator\Downloads\labels")

forward_name = "常用中药饮片识别与应用图谱"

for parent in image_parents.rglob("IMG*"):
    filename = parent.name
    parent.rename(f"{parent.parent}/{forward_name}-{filename}")

for parent in label_parents.rglob("IMG*"):
    filename = parent.name
    parent.rename(f"{label_parents}/{forward_name}-{filename}")
