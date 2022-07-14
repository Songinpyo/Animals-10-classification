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
      - Use 'Efficient_Net_v2', 'ConvNext_tiny' models
      - 비교적 경량화 된 모델 사용
      
  3. **K-Fold**
     - 5fold로 학습시켜 각 fold별 학습된 모델도 Essemble에 사용하였다.
      
  4. **Essemble**
      - Essemble the results of diffrent models using hard voting
      
## Training

Efficient_Net_v2와 ConvNext_tiny을 같은 Hyperparameter 에서 학습시켰다.


```
Hyperparameter : {
    'EPOCHS': 100,
    'BATCH_SIZE': 128,
    'LEARNING_RATE': 3e-4,
    'SEED': 41,
    'IMG_SIZE': 256
}
```

Validation set의 f1_score 가 최대일 때 모델을 저장하였다.

<details>
    <summary>자세히</summary>
    
Conv_fold1

![conv_flod0](https://user-images.githubusercontent.com/104220612/178905827-9c3a8378-e19b-4f14-84c1-20401db4f067.png)


Eff_v2_fold1

![effv2-fold0](https://user-images.githubusercontent.com/104220612/178905837-7a4dd503-4e27-4417-9b94-b6ec5c66d58c.png)


<!-- summary 아래 한칸 공백 두고 내용 삽입 -->

</details>


## Test

### ConvNext_tiny test f1 score
|Fold|test_f1|
|------|---|
|1|0.9853629492945964|
|2|0.9856762322420028|
|3|0.9858382581606346|
|4|0.9843437289411947|
|5|0.9861888643175929|

Best f1 score : 0.9861888643175929
<details>
    <summary>자세히</summary>
    
![image](https://user-images.githubusercontent.com/104220612/178904919-e801762a-2dfa-4c53-b70c-66b1df819c5d.png)

<!-- summary 아래 한칸 공백 두고 내용 삽입 -->

</details>

### Efficient_Net_v2 test f1 score
|Fold|test_f1|
|------|---|
|1|0.8927669712070969|
|2|0.8931974601492314|
|3|0.8937458721391404|
|4|0.8930224223796777|
|5|0.8950720707208372|

<details>
    <summary>자세히</summary>
    
![image](https://user-images.githubusercontent.com/104220612/178904974-b3969614-4478-40ef-b2f0-df35a354c231.png)

<!-- summary 아래 한칸 공백 두고 내용 삽입 -->

</details>

Efficient_Net_v2의 f1 score가 너무 안좋게 나와 ConvNext_tiny의 fold끼리만 essemble하기로 하였다.

Essemble에는 hardvoting을 채택하였다.
<details>
    <summary>자세히</summary>
    
![image](https://user-images.githubusercontent.com/104220612/178905062-dd6bc680-ea44-4ff7-9880-e3c1629e8044.png)

<!-- summary 아래 한칸 공백 두고 내용 삽입 -->

</details>

Essebled f1 score : **0.9863997420492107**

약 0.0002 f1 score가 상승하였음을 알 수 있다.


