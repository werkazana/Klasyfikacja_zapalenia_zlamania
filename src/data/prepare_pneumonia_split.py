import os, shutil, random

def make_split(src_root, dst_root, train_ratio=0.8, val_ratio=0.1):
    random.seed(27)

    classes = ["NORMAL", "PNEUMONIA"]
    splits = ["train", "val", "test"]

    for s in splits:
        for c in classes:
            os.makedirs(os.path.join(dst_root, s, c), exist_ok=True)

    for cls in classes:
        all_files = []
        for sub in ["train", "val", "test"]:
            class_path = os.path.join(src_root, sub, cls)
            for f in os.listdir(class_path):
                all_files.append(os.path.join(class_path, f))

        random.shuffle(all_files)
        n = len(all_files)

        train = all_files[: int(n*train_ratio)]
        val   = all_files[int(n*train_ratio): int(n*(train_ratio+val_ratio))]
        test  = all_files[int(n*(train_ratio+val_ratio)):]

        for group, name in [(train, "train"), (val, "val"), (test, "test")]:
            for f in group:
                shutil.copy2(f, os.path.join(dst_root, name, cls))
