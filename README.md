# Animals-10-classification
Classify animal pictures of 10 different categories taken from google images

## Dateset download
URL: https://www.kaggle.com/datasets/alessiocorrado99/animals10

  1. 위 URL에서 animal dataset을 다운로드하고 압축해제한다.
  2. 폴더 안에 아래 구조 처럼 "raw-img" 폴더만 추가한다. (translate.py 파일은 들고 올 필요 없습니다)
```
classfication
├── raw-img
│   ├── cane
│   ├── cavallo
│   └── ...
|
├── class.py
├── split.py
└── translate.py
```

## Preprocessing data
  1. cv2, numpy 설치
  ```
  pip install opencv-python, numpy
  ```
  
  2. split.py 실행
```
python split.py
```

verify : classfication 폴더 안에 train_img 와 test_img 폴더가 생기면 됩니다.

#### Dataset download, Preprocessing data code from [hsh-dev](https://github.com/hsh-dev/classfication)


## Main Idea

  1. **Augmentation**
      - Augmentate images using crop, rotate ...
  
  2. **Transfer learning**
      - Use 'Efficient_Net_v2', 'Efficient_Net_v1b5', 'ConvNext_tiny' models
      
  3. **Essemble**
      - Essemble the results of diffrent models using hard voting

