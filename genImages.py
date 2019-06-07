import os
import random
import pathlib
import shutil
import glob
import cv2
import numpy as np

RETURN_SUCCESS = 0
RETURN_FAILURE = -1

# Original Image Directory
IN_IMAGE_TRAIN = "./resize_img_train/*"
IN_IMAGE_TEST = "./resize_img_test/*"
# Output Directory
OUT_IMAGE_TRAIN = "./img_train"
OUT_IMAGE_TEST = "./img_test"

def load_name_images(image_path_pattern):
    name_images = []
    # 指定したパスパターンに一致するファイルの取得
    image_paths = glob.glob(image_path_pattern)
    # ファイルごとの読み込み
    for image_path in image_paths:
        path = pathlib.Path(image_path)
        # ファイルパス
        fullpath = str(path.resolve())
        print(f"画像ファイル（絶対パス）:{fullpath}")
        # ファイル名
        filename = path.name
        print(f"画像ファイル（名前）:{filename}")
        # 画像読み込み
        image = cv2.imread(fullpath)
        if image is None:
            print(f"画像ファイル[{fullpath}]を読み込めません")
            continue
        name_images.append((filename, image))
    return name_images

def scratch_image(image, use_flip=True, use_threshold=True, use_filter=True):
    # どの水増手法を利用するか（フリップ、閾値、平滑化）
    methods = [use_flip]
    # ぼかしに使うフィルターの作成
    # filter1 = np.ones((3, 3))
    # オリジナルの画像を配列に格納
    images = [image]
    # 水増手法の関数
    scratch = np.array([
        # フリップ処理
        lambda x: cv2.flip(x, 1),
        # 閾値処理
        #lambda x: cv2.threshold(x, 100, 255, cv2.THRESH_TOZERO)[1],
        # 平滑化処理
        #lambda x: cv2.GaussianBlur(x, (5, 5), 0),
    ])
    # 画像の水増
    doubling_images = lambda f, img: np.r_[img, [f(i) for i in img]]
    for func in scratch[methods]:
        images = doubling_images(func, images)
    return images

def delete_dir(dir_path, is_delete_top_dir=True):
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    if is_delete_top_dir:
        os.rmdir(dir_path)



def main():
    print("===================================================================")
    print("イメージ水増し OpenCV 利用版")
    print("指定した画像ファイルの水増し（フリップ＋閾値＋平滑化で8倍）を行います。")
    print("===================================================================")
    
    
    # ディレクトリの作成
    if not os.path.isdir(OUT_IMAGE_TRAIN):
        os.mkdir(OUT_IMAGE_TRAIN)
    # ディレクトリ内のファイル削除
    delete_dir(OUT_IMAGE_TRAIN, False)

    # ディレクトリの作成
    if not os.path.isdir(OUT_IMAGE_TEST):
        os.mkdir(OUT_IMAGE_TEST)
    # ディレクトリ内のファイル削除
    delete_dir(OUT_IMAGE_TEST, False)

    # 画像ファイルの読み込み
    name_images = load_name_images(IN_IMAGE_TRAIN)
   
    # 画像ごとの水増し
    for name_image in name_images:
        filename, extension = os.path.splitext(name_image[0])
        image = name_image[1]
        # 画像の水増し
        scratch_face_images = scratch_image(image)
        # 画像の保存
        for idx, image in enumerate(scratch_face_images):
            output_path = os.path.join(OUT_IMAGE_TRAIN, f"{filename}_{str(idx)}{extension}")
            print(f"出力ファイル（絶対パス）:{output_path}")
            cv2.imwrite(output_path, image)

    # 画像ファイルの読み込み
    name_images = load_name_images(IN_IMAGE_TEST)
   
    # 画像ごとの水増し
    for name_image in name_images:
        filename, extension = os.path.splitext(name_image[0])
        image = name_image[1]
        # 画像の水増し
        scratch_face_images = scratch_image(image)
        # 画像の保存
        for idx, image in enumerate(scratch_face_images):
            output_path = os.path.join(OUT_IMAGE_TEST, f"{filename}_{str(idx)}{extension}")
            print(f"出力ファイル（絶対パス）:{output_path}")
            cv2.imwrite(output_path, image)


    return RETURN_SUCCESS

if __name__ == "__main__":
    main()