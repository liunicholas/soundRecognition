import numpy as np

def main():
    dataPath = "/Users/nicholasliu/Documents/adhoncs/soundRecognition/spectrogramData/ALARM/alarm"
    with open(dataPath, 'rb') as f:
        numpyArray = np.load(f)

    print(np.shape(numpyArray))

main()
