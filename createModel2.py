import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from keras.models import Sequential
# from keras.utils.vis_utils import plot_model
# tensorflowのkerasはimport時にpydot_ng,pydotplusをimportするように記述されているが，
# keras(ver=2.2.0)はimport時，pydotしかimportするようにしか記述されていないため下記とする
from tensorflow.python.keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Input
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import CSVLogger

RETURN_SUCCESS = 0
RETURN_FAILURE = -1
# Outoput Model Only
OUTPUT_MODEL_ONLY = False
# Test Image Directory
TEST_IMAGE_DIR = "./img_test"
# Train Image Directory
TRAIN_IMAGE_DIR = "./img_train"
# Output Model Directory
OUTPUT_MODEL_DIR = "./model"
# Output Model File Name
OUTPUT_MODEL_FILE = "model.h5"
# Output Plot File Name
OUTPUT_PLOT_FILE = "model.png"
RESIZE = 64
EPOCS = 10
doGRAY = False
n_categories=5
def load_images(image_directory):
    image_file_list = []
    # 指定したディレクトリ内のファイル取得
    image_file_name_list = os.listdir(image_directory)
    print(f"対象画像ファイル数：{len(image_file_name_list)}")
    for image_file_name in image_file_name_list:
        # 画像ファイルパス
        image_file_path = os.path.join(image_directory, image_file_name)
        print(f"画像ファイルパス:{image_file_path}")
        # 画像読み込み
        image = cv2.imread(image_file_path)
        if image is None:
            print(f"画像ファイル[{image_file_name}]を読み込めません")
            continue
        image_file_list.append((image_file_name, image))
    print(f"読込画像ファイル数：{len(image_file_list)}")
    return image_file_list

def labeling_images(image_file_list):
    x_data = []
    y_data = []
    for idx, (file_name, image) in enumerate(image_file_list):
        # 画像をBGR形式からRGB形式へ変換
        if doGRAY:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # 画像配列（RGB画像）
        if doGRAY:
            x_data.append(np.expand_dims(image,axis=2))
        else:
            x_data.append(image)
            
        # ラベル配列（ファイル名の先頭2文字をラベルとして利用する）
        label = int(file_name[0:2])
        print(f"ラベル:{label:02}　画像ファイル名:{file_name}")
        y_data = np.append(y_data, label).reshape(idx+1, 1)
    x_data = np.array(x_data)
    print(f"ラベリング画像数：{len(x_data)}")
    return (x_data, y_data)

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

    # ディレクトリの作成
    if not os.path.isdir(OUTPUT_MODEL_DIR):
        os.mkdir(OUTPUT_MODEL_DIR)
    # ディレクトリ内のファイル削除
    delete_dir(OUTPUT_MODEL_DIR, False)

    num_classes = 5
    batch_size = 32


    # 学習用の画像ファイルの読み込み
    train_file_list = load_images(TRAIN_IMAGE_DIR)
    # 学習用の画像ファイルのラベル付け
    x_train, y_train = labeling_images(train_file_list)

    # plt.imshow(x_train[0])
    # plt.show()
    # print(y_train[0])

    # テスト用の画像ファイルの読み込み
    test_file_list = load_images(TEST_IMAGE_DIR)
    # 学習用の画像ファイルのラベル付け
    x_test, y_test = labeling_images(test_file_list)

    # plt.imshow(x_test[0])
    # plt.show()
    # print(y_test[0])

    # 画像とラベルそれぞれの次元数を確認
    print("x_train.shape:", x_train.shape)
    print("y_train.shape:", y_train.shape)
    print("x_test.shape:", x_test.shape)
    print("y_test.shape:", y_test.shape)

    # クラスラベルの1-hotベクトル化（線形分離しやすくする）
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # 画像とラベルそれぞれの次元数を確認
    print("x_train.shape:", x_train.shape)
    print("y_train.shape:", y_train.shape)
    print("x_test.shape:", x_test.shape)
    print("y_test.shape:", y_test.shape)


    if doGRAY:
        base_model = VGG16(include_top=False, weights='imagenet', input_shape=(RESIZE, RESIZE, 1))
    else:
        base_model = VGG16(include_top=False, weights='imagenet', input_shape=(RESIZE, RESIZE, 3))
    #add new layers instead of FC networks
    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(1024,activation='relu')(x)
    prediction=Dense(n_categories,activation='softmax')(x)
    model=Model(inputs=base_model.input,outputs=prediction)
    
    #fix weights before VGG16 14layers
    for layer in base_model.layers[:15]:
        layer.trainable=False

    # コンパイル
    model.compile(optimizer=SGD(lr=0.0001,momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # サマリーの出力
    model.summary()

    # モデルの可視化
    plot_file_path = os.path.join(OUTPUT_MODEL_DIR, OUTPUT_PLOT_FILE)
    plot_model(model, to_file=plot_file_path, show_shapes=True)

    if OUTPUT_MODEL_ONLY:
        # 学習
        model.fit(x_train, y_train, batch_size=batch_size, epochs=EPOCS)
    else:
        # 学習+グラフ
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=EPOCS,
                            verbose=1, validation_data=(x_test, y_test))

        # 汎化精度の評価・表示
        test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
        print(f"validation loss:{test_loss}\r\nvalidation accuracy:{test_acc}")

        # acc（精度）, val_acc（バリデーションデータに対する精度）のプロット
        plt.plot(history.history["acc"], label="acc", ls="-", marker="o")
        plt.plot(history.history["val_acc"], label="val_acc", ls="-", marker="x")
        plt.title('model accuracy')
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend(loc="best")
        plt.show()

        # 損失の履歴をプロット
        plt.plot(history.history['loss'], label="loss", ls="-", marker="o")
        plt.plot(history.history['val_loss'], label="val_loss", ls="-", marker="x")
        plt.title('model loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc='lower right')
        plt.show()

    # モデルを保存
    model_file_path = os.path.join(OUTPUT_MODEL_DIR, OUTPUT_MODEL_FILE)
    model.save(model_file_path)

    return RETURN_SUCCESS

if __name__ == "__main__":
    main()