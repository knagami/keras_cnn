import os
import glob
import pathlib
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

# Original Image Directory
IN_IMAGE_TRAIN = "./img_train/*"
IN_IMAGE_TEST = "./img_test/*"
# Output Directory
OUT_IMAGE_TRAIN = "./img_train"
OUT_IMAGE_TEST = "./img_test"

def draw_images(generator, x, dir_name, fname):
    # 出力ファイルの設定
    save_name = fname
    g = generator.flow(x, batch_size=1, save_to_dir=dir_name, save_prefix=fname, save_format='jpg')

    # 1つの入力画像から何枚拡張するかを指定
    # g.next()の回数分拡張される
    for i in range(2):
        bach = g.next()
        
def delete_dir(dir_path, is_delete_top_dir=True):
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    if is_delete_top_dir:
        os.rmdir(dir_path)

if __name__ == '__main__':

    # 拡張する際の設定
    generator = ImageDataGenerator(
                    rotation_range=30 # 90°まで回転
                    )

    # 拡張する画像群の読み込み
    #images = glob.glob(os.path.join('./', "*.jpg"))
    images = glob.glob(IN_IMAGE_TRAIN)
    for i in range(len(images)):
        img = load_img(images[i])
        path = pathlib.Path(images[i])
        print(path.name)
        # 画像を配列化して転置a
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        # 画像の拡張
        draw_images(generator, x, OUT_IMAGE_TRAIN, path.name)
        
    # 拡張する画像群の読み込み
    #images = glob.glob(os.path.join('./', "*.jpg"))
    images = glob.glob(IN_IMAGE_TEST)
    for i in range(len(images)):
        img = load_img(images[i])
        path = pathlib.Path(images[i])
        print(path.name)
        # 画像を配列化して転置a
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        # 画像の拡張
        draw_images(generator, x, OUT_IMAGE_TEST, path.name)