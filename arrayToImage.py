from PIL import Image
import PIL
import numpy as np
import os
import tensorflow as tf

dataPath = "./spectrogramData"
newPath  ="./pictureData"

def main():
    folders = os.listdir(dataPath)
    index = folders.index(".DS_Store")
    folders.pop(index)
    print(folders)
    for folder in folders:
        print("new folder")
        if os.path.isdir(f"{dataPath}/{folder}"):
            for filename in os.listdir(f"{dataPath}/{folder}"):
                print("new file")
                filePath = f"{dataPath}/{folder}/{filename}"
                print(filePath)

                with open(filePath, 'rb') as f:
                    numpyArray = np.load(f)
                    print(np.shape(numpyArray))

                    stacked_array = np.stack((numpyArray,)*3, axis=-1)
                    print(np.shape(stacked_array))

                    img = tf.keras.preprocessing.image.array_to_img(stacked_array)

                    if not os.path.exists(f"{newPath}/{folder}"):
                        os.makedirs(f"{newPath}/{folder}")

                    img.save(f"{newPath}/{folder}/{filename}", "PNG")

main()
