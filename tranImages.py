import os
import numpy as np
import shutil
import cv2
from pprint import pprint

# from keras.utils.vis_utils import plot_model
# tensorflowのkerasはimport時にpydot_ng,pydotplusをimportするように記述されているが，
# keras(ver=2.2.0)はimport時，pydotしかimportするようにしか記述されていないため下記とする
from tensorflow.python.keras.utils.vis_utils import plot_model


RETURN_SUCCESS = 0
RETURN_FAILURE = -1
# Train Image Directory
IN_IMAGE_DIR= "./shoes_img"
OUT_IMAGE_TRAIN= "./shoes_img_train"
OUT_IMAGE_TEST = "./shoes_img_test"
OUT_IMAGE_RINKAKU = "./shoes_img_rinkaku"
RESIZE=128

def load_images(image_directory):
    i = 0
    # 指定したディレクトリ内のファイル取得
    image_file_name_list = os.listdir(image_directory)
    pprint(image_file_name_list)
    for image_file_name in image_file_name_list:
        if image_file_name != ".DS_Store":
            print(image_file_name);
            image_file_name_list2 = os.listdir(image_directory + "/" + image_file_name)
            j = 0
            for image_file_name2 in image_file_name_list2:
                if image_file_name2 != ".DS_Store":
                    print(image_file_name2);
                    in_file_path = IN_IMAGE_DIR + "/" + image_file_name + "/"  + image_file_name2
                    out_file_path = OUT_IMAGE_TRAIN + "/0" + str(i) + image_file_name2
                    out_file_path2 = OUT_IMAGE_RINKAKU + "/" + image_file_name2
                    if j < 3:
                        out_file_path = OUT_IMAGE_TEST + "/0" + str(i) + image_file_name2
                    doRinkaku(in_file_path,out_file_path,out_file_path2)
                    doSquare(out_file_path2,out_file_path)
                    doScale(out_file_path,out_file_path)

                    #doGlay(out_file_path,out_file_path)
                    j=j + 1
            i=i + 1

def doGlay(in_file_path,out_file_path ):
    # 画像ファイルのグレースケール化
    img = cv2.imread(in_file_path)
    image_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img2 = cv2.resize(image_gs ,64, 64)
    cv2.imwrite(out_file_path, image_gs)
    
def doSquare(in_file_path,out_file_path):
    img = cv2.imread(in_file_path)
    tmp = img[:, :]
    height, width = img.shape[:2]
    if(height > width):
        size = height
        limit = width
    else:
        size = width
        limit = height
    start = int((size - limit) / 2)
    fin = int((size + limit) / 2)
    new_img = cv2.resize(np.zeros((2, 1, 3), np.uint8), (size, size))
    new_img.fill(255)
    if(size == height):
        new_img[:, start:fin] = tmp
    else:
        new_img[start:fin, :] = tmp
    cv2.imwrite(out_file_path, new_img)
    
def doScale(in_file_path,out_file_path):
    img = cv2.imread(in_file_path)
    image_gs = cv2.resize(img, (RESIZE,RESIZE))
    cv2.imwrite(out_file_path, image_gs)
                
def doRinkaku(in_file_path,out_file_path,out_file_path2):
    # original（輪郭記述用）
    img_org = cv2.imread(in_file_path)
    # グレースケール化
    img_tmp = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
    # 2値化
    ret, img_tmp = cv2.threshold(img_tmp,250, 256, cv2.THRESH_BINARY_INV) # 2値化type
    #ret, img_tmp = cv2.threshold(img_tmp, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #img_tmp = cv2.adaptiveThreshold(img_tmp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)
    # 黒塗り画像作成
    ret, img_tmp2 = cv2.threshold(img_tmp,255,256,cv2.THRESH_BINARY) 
    # 境界線探索
    # - 第2引数:
    #   - cv2.RETR_EXTERNAL は最外周のみ探索
    #   - cv2.RETR_TREE     は全境界(輪郭? 等高線?)を探索
    # - 返り値:
    #   - contours : 探索された境界
    #   - hierarchy: 境界が複数ある場合の階層
    contours, hierarchy = cv2.findContours(img_tmp,
                                                cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE )

    contours.sort(key=cv2.contourArea, reverse=True)
    i  = 0
    for contour in contours:
        arclen = cv2.arcLength(contour,
                               True) # 対象領域が閉曲線の場合、True
        approx = cv2.approxPolyDP(contour,
                                  0.005*arclen,  # 近似の具合?
                                  True)
        areas = cv2.contourArea(contour)
        max_index = np.argmax(areas)
        cnt=contours[max_index]
        x,y,w,h = cv2.boundingRect(cnt)
        clipped = img_org[y:y+h, x:x+w]
        cv2.imwrite(out_file_path2, clipped)
        break
 
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
    print("モデル学習 Keras 利用版")
    print("指定した画像ファイルをもとに学習を行いモデルを生成します。")
    print("===================================================================")

    # ディレクトリ内のファイル削除
    delete_dir(OUT_IMAGE_TRAIN, False)
    delete_dir(OUT_IMAGE_TEST, False)
    delete_dir(OUT_IMAGE_RINKAKU, False)

    load_images(IN_IMAGE_DIR)
   
    return RETURN_SUCCESS

if __name__ == "__main__":
    main()