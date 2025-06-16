import os

def get_all_relative_paths(root_dir):
    """
    Walks through the directory and returns a set of all file and folder paths
    relative to the root_dir.
    """
    all_paths = set()
    for base, dirs, files in os.walk(root_dir):
        rel_base = os.path.relpath(base, root_dir)
        for name in files + dirs:
            path = os.path.normpath(os.path.join(rel_base, name))
            all_paths.add(path)
    return all_paths

def compare_dirs_recursive(dir1, dir2):
    paths1 = get_all_relative_paths(dir1)
    paths2 = get_all_relative_paths(dir2)

    only_in_dir1 = paths1 - paths2
    only_in_dir2 = paths2 - paths1

    print(f"\nItems only in {dir1}:")
    print('number',len(only_in_dir1))
    for item in sorted(only_in_dir1):
        # print("  ", item)
        if not ("Edge-Map" in item or "Flattened" in item):
            print(" ",item)

    print(f"\nItems only in {dir2}:")
    for item in sorted(only_in_dir2):
        print("  ", item)

if __name__ == "__main__":
    dir1 ="d:\cleaning_GUI_annotated_Data\Cirrus_OCT_Imaging_Data"
    dir2 = "d:\cleaning_GUI_annotated_Data\denoised_data"

    if not os.path.isdir(dir1) or not os.path.isdir(dir2):
        print("One or both of the provided paths are not valid directories.")
    else:
        compare_dirs_recursive(dir1, dir2)
