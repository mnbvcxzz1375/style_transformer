import os
import glob


def clear_folder(folder_path):
    # 获取文件夹中所有文件的路径
    files = glob.glob(os.path.join(folder_path, '*'))

    # 删除每个文件
    for file_path in files:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


def is_film_empty(folder_path):
     if not os.path.exists(folder_path):
         print(f"文件夹 '{folder_path}' 不存在。")
         return None

     contents = os.listdir(folder_path)

     if not contents:
         return True
     else:
         return False



