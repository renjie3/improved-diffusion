import blobfile as bf

import os
# os.remove("demofile.txt")

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

all_files = _list_image_files_recursively("/localscratch/renjie/cifar_train_20000")

class_names = [bf.basename(path).split("_")[0] for path in all_files]
sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
adv_output_classes = [sorted_classes[x] for x in class_names]

for i, file in enumerate(all_files):
    if i % 10 >= 4:
        os.remove(file)
