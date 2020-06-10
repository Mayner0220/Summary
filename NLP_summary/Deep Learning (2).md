# Deep Learning (2)

Source: https://wikidocs.net/22882, https://wikidocs.net/24958, https://wikidocs.net/24987, https://wikidocs.net/36033, https://wikidocs.net/37406, https://wikidocs.net/61374, https://wikidocs.net/61375, https://wikidocs.net/32105, https://wikidocs.net/38861, https://wikidocs.net/49071, https://wikidocs.net/45609

---

### 4. 과적합(Overfitting)을 막는 방법

학습 데이터에 모델이 과적합되는 현상은 모델의 성능을 떨어트리는 주요 이슈이다.
모델이 과적합되면 훈련 데이터에 대한 정확도는 높을지라도, 새로운 데이터, 즉, 검증 데이터나 테스트 데이터에 대해서는 제대로 동작하지 않는다.
이는 모델이 학습 데이터를 불필요할정도로 과하게 암기하여 훈련 데이터에 포함된 노이즈까지 학습한 상태라고 해석할 수 있다.
이번에는 모델의 과적합을 막을 수 있는 여러가지 방법에 대해서 논의해보자.

---

### 4.1 데이터의 양을 늘리기

모델은 데이터의 양이 적을 경우, 해당 데이터의 특정 패턴이나 노이즈까지 쉽게 암기하기 되므로 과적합 현상이 발생할 확률이 늘어난다.
그렇기 때문에 데이터의 양을 늘릴 수록 모델은 데이터의 일반적인 패턴을 학습하여 과적합을 방지할 수 있다.

만약, 데이터의 양이 적을 경우에는 의도적으로 기존의 데이터를 조금씩 변혀앟고 추가하여 데이터의 양을 늘리기도 하는데 이를 데이터 증식 또는 증강(Data Augmentation)이라고 한다.
이미지의 경우에는 데이터 증식이 많이 사용되는데 이미지를 돌리거나 노이즈를 추가하고, 일부분 수정하는 등으로 데이터를 증식시킨다.

---

### 4.2 모델의 복잡도 줄이기

인공 신경망의 복잡도는 은닉층(hidden layer)의 수나 매개변수의 수 등으로 결정된다.
과적합 현상이 포착되었을 때, 인공 신경망 모델에 대해서 할 수 있는 한 가지 조치는 인공 신경망의 복잡도를 줄이는 것이다.

- 인공 신경망에서는 모델에 있는 매개변수들의 수를 모델의 수용력(capacity)이라고 하기도 한다.

---

### 4.3 가중치 규제(Regularization) 적용하기

복잡한 모델이 간단한 모델보다 과적합될 가능성이 높다.
그리고 간단한 모델은 적은 수의 매개변수를 가진 모델을 말한다.
복잡한 모델을 좀 더 간단하게 하는 방법으로 가중치 규제가 있다.

- L1 규제: 가중치 w들의 절대값 합계를 비용 함수에 추가한다. (L1 노름이라고도 한다.)
- L2 규제: 모든 가중치 w들의 제곱합을 비용 함수에 추가한다. (L2 노름이라고도 한다.)

L1 규제는 기존의 비용 함수에 모든 가중치에 대해서 λ∣w∣를 더한 값을 바용함수로 하고, L2 규제는 기존의 비용 함수에 모든 가중치에 대해서 1/2λw^2 를 더한 값을 비용 함수로 한다.
λ는 규제의 강도를 정하는 하이퍼파라미터이다.
λ가 크다면 모델이 훈련 데이터에 대해서 적합한 매개 변수를 찾는 것보다 규제를 위해 추가된 항들을 작게 유지하는 것을 우선한다는 의미가 된다.

이 두 식 모두 비용 함수를 최소화하기 위해서는 가중치 w들의 값이 작아져야 한다는 특징이 있다.
L1 규제로 예를 들어보자.
L1 규제를 사용하면 비용 함수가 최소가 되게 하는 가중치와 편향을 찾는 동시에 가중치들의 절대값의 합도 최소가 되어야한다.
이렇게 되면, 가중치 w의 값들은 0 또는 0에 가까이 작아져야 하므로 어떤 특성들은 모델을 만들 때 거의 사용되지 않게 된다.

예를 들어 H(x)=w1x1+w2x2+w3x3+w4x4라는 수식이 있다고 해보자.
여기에 L1 규제를 사용하였더니, w3의 값이 0이 되었다고 해봅시다. 이는 x3 특성은 사실 모델의 결과에 별 영향을 주지 못하는 특성임을 의미한다.

L2 규제는 L1 규제와는 달리 가중치들의 제곱을 최소화하므로 w의 값이 완전히 0이 되기보다는 0에 가까워지기는 경향을 뛴다.
L1 규제는 어떤 특성들이 모델에 영향을 주고 있는지를 정확히 판단하고자 할 때 유용하다.
만약, 이런 판단이 필요없다면 경험적으로는 L2 규제가 더 잘 동작하므로 L2 규제를 더 권장한다.
인공 신경망에서 L2 규제는 가중치 감쇠(weight decay)라고도 부른다.

---

### 4.4 드롭아웃(Dropout)

드롭아웃은 학습 과정에서 신경망의 일부를 사용하지 않는 방법이다.
예를 들어 드롭아웃의 비율을 0.5로 한다면 학습 과정마다 랜덤으로 절반의 뉴런을 사용하지 않고, 절반의 뉴런만을 사용한다.

![img](https://wikidocs.net/images/page/60751/%EB%93%9C%EB%A1%AD%EC%95%84%EC%9B%83.PNG)

드롭아숭은 신경망 학습 시에만 사용하고, 예측 시에는 사용하지 않는 것이 일반적이다.
학습 시에 인공 신경망이 특정 뉴런 또는 특정 조합에 너무 의존적이게 되는 것을 방지해주고, 매번 랜덤 선택으로 뉴런들을 사용하지 않으므로 서로 다른 신경망들을 앙상블하여 사용하는 것 같은 효과를 내어 과적합을 방지한다.

케라스에서는 다음과 같은 방법으로 드롭아웃을 모델에 추가할 수 있다. 

```python
model = Sequential()
model.add(Dense(256, input_shape=(max_words,), activation='relu'))
model.add(Dropout(0.5)) # 드롭아웃 추가. 비율은 50%
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5)) # 드롭아웃 추가. 비율은 50%
model.add(Dense(num_classes, activation='softmax'))
```

---

### 5. 기울기 소실(Gradient Vanishing)과 폭주(Exploding)

깊은 인공 신경망을 학습하다보면 역전파 과정에서 입력층으로 갈 수록 기울기(Gradient)가 점차적으로 작아지는 현상이 발생할 수 있다.
입력층에 가까운 충돌에서 가중치들이 업데이트가 제대로 되지 않으면 결국 최적의 모델을 찾을 수 없게 된다.
이를 기울기 소실(Gradient)(Gradient)이라고 한다.

반대의 경우도 있다.
기울기가 점차 커지더니 가중치들이 비정상적으로 큰 값이 되면서 결국 발산되기도 한다.
이를 기울기 폭주(Gradient Exploding)이라고 하며, 뒤에서 배울 순환 신경망(Recurrent Neural Network, RNN)에서 발생할 수 있다.

---

### 5.1 ReLU와 ReLU의 변형들

앞에서 배운 내용을 간단하게 복습해보자.
시그모이드 함수를 사용하면 입력의 절대값이 클 경우에 시그모이드 함수의 출력값이 0 또는 1에 수렴하면서 기울기가 0에 가까워진다.
그래서 역전파 과정에서 전파 시킬 기울기가 점차 사라져서 입력층 방향으로 갈 수록 제대로 역전파가 되지 않는 기울기 소실 문제가 발생할 수 있다.

기울기 소실을 완화하는 가장 간단한 방법은 은닉층의 활성화 함수로 시그모이드나 하이퍼볼릭탄젠트 함수 대신에 ReLU나 ReLU의 변형 함수와 같은 Leaky ReLU를 사용하는 것이다.

- 은닉층에서는 시그모이드 함수를 사용하지 마세요.
- Leaky ReLU를 사용하면 모든 입력값에 대해서 기울기가 0에 수렴하지 않아 죽은 ReLU 문제를 해결한다.
- 은닉층에서는 ReLU나 Leaky ReLU와 같은 ReLU 함수의 변형들을 사용하자.

---

### 5.2 그래디언트 클리핑(Gradient Clipping)

그래디언트 클리핑은 말 그대로  기울기 값을 자르는 것을 의미한다.
기울기 폭주를 막기 위해 임계값을 넘지 않도록 값을 자른다.
다시 말해서 임계치만큼 크기를 감소시킨다.
이는 RNN에서 유용하다.
RNN은 BPTT에서 시점을 역해와면서 기울기를 구하는데, 이때 기울기가 너무 커질 수 있기 때문이다.
케라스에서는 다음과 같은 방법으로 그래디언트 클리핑을 수행한다.

```python
from tensorflow.keras import optimizers
Adam = optimizers.Adam(lr=0.0001, clipnorm=1.)
```

---

### 5.3 가중치 초기화(Weight initialization)

같은 모델을 훈련시키더라도 가중치가 초기에 어떤 값을 가졌느냐에 따라서 모델의 훈련 결과가 달라지기도 한다.
다시 말해 가중치 초기화만 적절히 해줘도 기울기 소실 문제과 같은 문제를 완화시킬 수 있다.

1. 세이비어 초기화(Xavier Initialization)

   - 논문 : http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

   2010년 세이비어 글로럿과 요슈아 벤지오는 가중치 초기화가 모델에 미치는 영향을 분석하여 새로운 초기화 방법을 제안했다.
   이 초기화 방법은 제안한 사람의 이름을 따서 세이비어 초기화(Xavier Initialization) 또는 글로럿 초기화(Glorot Initialization)라고 한다.

   이 방법은 균등 분포(Uniform Distribution) 또는 정규 분포(Normal Distribution)로 초기화 할 때 두 가지 경우로 나뉘며, 이전 층의 뉴런 개수와 다음 층의 뉴런 개수를 가지고 식을 세운다.
   이전 층의 뉴런의 개수를 
   $$
   n_{in}
   $$
   , 다음 층의 뉴런의 개수를 
   $$
   n_{out}
   $$
   이라고 해보자.

   글로럿과 벤지오의 논문에서는 균등 분포를 사용하여 가중치를 초기화할 경우 다음과 같은 균등 분포 범위를 사용하라고 한다.
   $$
   W \sim Uniform(-\sqrt{\frac{6}{ {n}_{in} + {n}_{out} }}, +\sqrt{\frac{6}{ {n}_{in} + {n}_{out} }})
   $$
   다시 말해 
   $$
   \sqrt{\frac{6}{ {n}_{in} + {n}_{out} }}
   $$
   를 m이라고 하였을 때, -m과 +m 사이의 균등 분포를 의미한다.

   정규 분포로 초기화할 경우에는 평균이 0이고, 표준 편차 σ가 다음을 만족하도록 한다.
   $$
   σ=\sqrt{\frac { 2 }{ { n }_{ in }+{ n }_{ out } } }
   $$
   세이비어 초기화는 여러 층의 기울기 분산 사이에 균형을 맞춰서 특정 층이 너무 주목을 받거나 다른 층이 뒤쳐지는 것을 막는다.
   그런데 세이비어 초기화는 시그모이드 함수나 하이퍼볼릭 탄젠트 함수와 같은 S자 형태인 활성화 함수와 함께 사용할 경우에는 좋은 성능을 보이지만, ReLU와 함께 사용할 경우에는 성능이 좋지 않다.
   ReLU 함수 또는 ReLU의 변형 함수들을 활성화 함수로 사용할 경우에는 다른 초기화 방법을 사용하는 것이 좋은데, 이를 He 초기화(He initialization)라고 한다.

2. He 초기화(He initialization)

   - 논문 : https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf

   He 초기화는 세이비어 초기화와 유사하게 정규 분포와 균등 분포 두 가지 경우로 나뉜다.
   다만, He 초기화는 세이비어 초기화와 다르게 다음 층의 뉴런의 수를 반영하지 않는다.
   전과 같이 이전 층의 뉴런의 개수를 
   $$
   n_{in}
   $$
   이라고 해보자.

   He 초기화는 균등 분포로 초기화 할 경우에는 다음과 같은 균등 분포 범위를 가지도록 한다.
   $$
   σ=\sqrt{\frac { 2 }{ { n }_{ in } } }
   $$

   - 시그모이드 함수나 하이퍼볼릭탄젠트 함수를 사용할 경우에는 세이비어 초기화 방법이 효율적이다.
   - ReLU 계열 함수를 사용할 경우에는 He 초기화 방법이 효율적이다.
   - ReLU + He 초기화 방법이 좀 더 보편적이다.

---

### 5.4 배치 정규화(Batch Normalization)

ReLU 계열의 함수와 He 초기화를 사용하는 것만으로도 어느 정도 기울기 소실과 폭주를 완화시킬 수 있지만, 이 두 방법을 사용하더라도 훈련 중에 언제든 다시 발생할 수 있따.
기울기 소실이나 폭주를 예방하는 또 다른 방법은 배치 정규화(Batch Normalization)이다.
배치 정규화는 인공 신경망의 각 층에 들어가는 입력을 평균과 분산으로 정규화하여 학습을 효율적으로 만든다.

1. 내부 공변량 변화(Internal Covariate Shift)
   배치 정규화를 이해하기 위해서는 내부 공변향 변화(Internal Covariate Shift)를 이해할 필요가 있다.
   내부 공변향 변화란 학습 과정에서 층 별로 입력 데이터 분포가 달라지는 현상을 말한다.
   이전 층들의 학습에 의해 이전 층의 가중치 값이 바뀌게 되면, 현재 층에 전달되는 입력 데이터의 분포가 현재 층이 학습했던 시점의 분포와 차이가 발생한다.
   배치 정규화를 제안한 논문에서는 기울기 소실/폭주 등의 딥러닝 모델의 불안전성ㅇ이 층마다 입력의 분포가 달라지기 때문이라고 주장한다.

   - 공변량 변화는 훈련 데이터의 분포와 테스트 데이터의 분포가 다른 경우를 의미한다.
   - 내부 공변량 변화는 신경망 층 사이에서 발생하는 입력 데이터의 분포 변화를 의미한다.

2. 배치 정규화(Batch Normalization)
   배치 정규화는 표현 그대로 한 번에 들어오는 배치 단위로 정규화하는 것을 말한다.
   배치 정규화는 각 층에서 활성화 함수를 통과하기 전에 수행된다.
   배치 정규화를 요약하면 다음과 같다.
   입력에 대해 평균을 0으로 만들고, 정규화를 한다.
   그리고 정규화된 데이터에 대해서 스케일과 시프트를 수행한다.
   이 때 두 개의 매개변수 γ와 β를 사용하는데, γ는 스케일을 위해 사용하고, β는 시프트를 하는 것에 사용하며 다음 레이어에 일정한 범위의 값들만 전달되게 한다.

   배치 정규화의 수식은 다음과 같다.
   아래에서 BN은 배치 정규화를 의미한다.

   - Input: 미니 배치 
     $$
     B = \{{x}^{(1)}, {x}^{(2)}, ..., {x}^{(m)}\}
     $$

   - Output: 
     $$
     y^{(i)} = BN_{γ, β}(x^{(i)})
     $$

   $$
   μ_{B} ← \frac{1}{m} \sum_{i=1}^{m} x^{(i)} \text{ # 미니 배치에 대한 평균 계산}
   $$

$$
σ^{2}_{B} ← \frac{1}{m} \sum_{i=1}^{m} (x^{(i)} - μ_{B})^{2}\text{ # 미니 배치에 대한 분산 계산}
$$

$$
\hat{x}^{(i)} ← \frac{x^{(i)} - μ_{B}}{\sqrt{σ^{2}_{B}+ε}}\text{ # 정규화}
$$
$$
y^{(i)} ← γ\hat{x}^{(i)} + β = BN_{γ, β}(x^{(i)}) \text{ # 스케일 조정(γ)과 시프트(β)를 통한 선형 연산}
$$

- m은 미니 배치에 있는 샘플의 수
   - μB는 미니 배치 B에 대한 평균
   - σB는 미니 배치 B에 대한 표준편차
   - x^(i)은 평균이 0이고 정규화 된 입력 데이터
   - ε은 σ2가 0일 때, 분모가 0이 되는 것을 막는 작은 양수. 보편적으로 10^(−5)
   - γ는 정규화 된 데이터에 대한 스케일 매개변수로 학습 대상
   - β는 정규화 된 데이터에 대한 시프트 매개변수로 학습 대상
   - y(i)는 스케일과 시프트를 통해 조정한 BN의 최종 결과
   

배치 정규화는 학습 시 배치 단위의 평균과 분산들을 차례대로 받아 이동 평균과 이동 분산을 저장해놓았다가 테스트 할 때는 해당 배치의 평균과 분산을 구하지 않고 구해놓았던 평균과 분산으로 정규화를 한다.

- 배치 정규화를 사용하면 시그모이드 함수나 하이퍼볼릭탄젠트 함수를 사용하더라도 기울기 소실 문제가 크게 개선된다.
   - 가중치 초기화에 훨씬 덜 민감해진다.
   - 훨씬 큰 학습률을 사용할 수 있어 학습 속도를 개선시킨다.
   - 미니 배치마다 평균과 표준편차를 계산하여 사용하므로 훈련 데이터에 일종의 잡음 주입의 부수 효과로 과적합을 방지하는 효과도 냅니다. 다시 말해, 마치 드롭아웃과 비슷한 효과를 냅니다. 물론, 드롭 아웃과 함께 사용하는 것이 좋다.
   - 배치 정규화는 모델을 복잡하게 하며, 추가 계산을 하는 것이므로 테스트 데이터에 대한 예측 시에 실행 시간이 느려집니다. 그래서 서비스 속도를 고려하는 관점에서는 배치 정규화가 꼭 필요한지 고민이 필요하다.
   - 배치 정규화의 효과는 굉장하지만 내부 공변량 변화때문은 아니라는 논문도 있다.
     https://arxiv.org/pdf/1805.11604.pdf
   
3. 배치 정규화의 한계
   배치 정규하는 뛰어난 방법이지만 몇 가지 한계가 존재한다.

   1. 미니 배치 크기에 의존적이다.
      배치 정규화는 너무 작은 배치 크기에서는 잘 작동하지 않을 수 있다.
      단적으로 배치 크기를 1로 하게되면 분산은 0이 된다.
      작은 미니 배치에서는 배치 정규화의 효과가 극단적으로 작용되어 훈련에 악영향을 줄 수 있다.
      배치 정규화를 적용할 때는 작은 미니 배치보다는 크기가 어느 정도 되는 미니 배치에서 하는 것이 좋다.
      이 처럼 배치 정규화는 배치 크기에 의존적인 면이 있다.
   2. RNN에 적용하기 어렵다.
      뒤에서 배우지만, RNN은 각 시점(time step)마다 다른 통계치를 가진다.
      이는 RNN에 배치 정규화를 적용하기 위한 몇 가지 논문이 제시되어 있지만, 여기서는 이를 소개하는 대신 배치 크기(layer normalization)라는 방법을 소개하고자 한다.

---

### 5.5 층 정규화(Layer Normalization)

층 정규화를 이해하기에 앞서 배치 정규화를 시각화해보자.
다음은 m이 3이고, 특성의 수가 4일 때의 배치 정규화를 보여준다.
미니 배치란 동일한 특성(feature) 개수들을 가진 다수의 샘플들을 의미함을 상기하자.

![img](https://wikidocs.net/images/page/61375/%EB%B0%B0%EC%B9%98%EC%A0%95%EA%B7%9C%ED%99%94.PNG)

반면, 층 정규화는 다음과 같다.

![img](https://wikidocs.net/images/page/61375/%EC%B8%B5%EC%A0%95%EA%B7%9C%ED%99%94.PNG)

---

### 6.케라스 훑어보기

케라스는 유저가 손쉽게 딥 러닝을 구현할 수 있도록 도와두는 상위 레벨의 인터페이스이다.
케라스를 사용하면 딥 러닝을 쉽게 구현할 수 있다.

케라스의 모든 기능들을 열거하는 것만으로도 하나의 책 분량이고, 여기서 전부 다룰 수는 없다.
가장 좋은 방법은 케라스 공식문서를 참고하는 것이다.

- 케라스 공식 문서: https://keras.io/

---

### 6.1 전처리(Processing)

Tokenizer(): 토큰화와 정수 인코딩(단어에 대한 인덱싱)을 위해 사용된다.

```python
from tensorflow.keras.preprocessing.text import Tokenizer
t  = Tokenizer()
fit_text = "The earth is an awesome place live"
t.fit_on_texts([fit_text])

test_text = "The earth is an great place live"
sequences = t.texts_to_sequences([test_text])[0]

print("sequences : ",sequences) # great는 단어 집합(vocabulary)에 없으므로 출력되지 않는다.
print("word_index : ",t.word_index) # 단어 집합(vocabulary) 출력
```

```python
sequences :  [1, 2, 3, 4, 6, 7]
word_index :  {'the': 1, 'earth': 2, 'is': 3, 'an': 4, 'awesome': 5, 'place': 6, 'live': 7}
```

pad_sequence(): 전체 훈련 데이터에서 각 샘플의 길이는 서로 다를 수 있다.
또는 각 문서 또는 각 문장은 단어의 수가 제각각이다.
모델의 입력으로 사용하려면 모든 샘플의 길이를 동일하게 맞춰어야 할 때가 있다.
이를 NLP에서는 패딩(padding) 작업이라고 하는데, 보통 숫자 0을 넣어서 길이가 다른 샘플들의 길이를 맞춰보자.
케라스에서는 pad_sequence()를 사용한다.
pad_sequence()는 정해준 길이보다 길이가 긴 샘플은 값을 일부 자르고, 정해준 길이보다 길이가 짧은 샘플은 값을 0으로 채운다.

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_sequences([[1, 2, 3], [3, 4, 5, 6], [7, 8]], maxlen=3, padding='pre')
# 전처리가 끝나서 각 단어에 대한 정수 인코딩이 끝났다고 가정하고, 3개의 데이터를 입력으로 합니다.
```

```python
array([[1, 2, 3],
       [4, 5, 6],
       [0, 7, 8]], dtype=int32)
```

첫번째 인자 = 패딩을 진행할 데이터
maxlen = 모든 데이터에 대해서 정규화 할 길이
padding = 'pre'를 선택하면 앞에 0을 채우고 'post'를 선택하면 뒤에 0을 채움.

---

### 6.2 워드 임베딩(Word Embedding)

워드 임베딩이란 텍스트 내의 단어들을 밀집 벡터(dense vector)로 만드는 것이다.
밀집 벡터가 무엇일까.
이미 배운 개념인 원-핫 벡터와 비교해보자.
원-핫 벡터는 대부분이 0의 값을 가지고, 단 하나의 1의 값을 가지는 벡터였다.
또한 벡터의 차원이 대체적으로 크다는 성질을 가졌다.
원-핫 벡터의 예는 다음과 같다.

- Ex) [0 1 0 0 0 0 ... 중략 ... 0 0 0 0 0 0 0] # 차원이 굉장히 크면서 대부분의 값이 0

대부분의 값이 0인 이러한 벡터를 희소 벡터(sparse vector)라고 한다.
원-핫 벡터는 희소 벡터의 예이다.
원-핫 벡터는 단어의 수만큼 벡터의 차원을 가지며 단어 간 유사도가 모두 동일하다는 단점이 있다.
반면, 희소 벡터와 표기상으로도 의미상으로도 반대인 벡터가 있다.
대부분의 값이 실수이고, 상대적으로 저차원인 밀집 벡터(dense vector)이다.
아래는 밀집 벡터의 예이다.

- Ex) [0.1 -1.2 0.8 0.2 1.8] # 상대적으로 저차원이며 실수값을 가짐

간단히 표로 정리하면 아래와 같다.

|     -     |        원-핫 벡터        |       임베딩 벡터        |
| :-------: | :----------------------: | :----------------------: |
|   차원    | 고차원(단어 집합의 크기) |          저차원          |
| 다른 표현 |     희소 벡터의 일종     |     밀집 벡터의 일종     |
| 표현 방법 |           수동           | 훈련 데이터로부터 학습함 |
| 값의 타입 |          1과 0           |           실수           |

단어를 원-핫 벡터로 만드는 과정을 원-핫 인코딩이라고 한다.
이와 대비적으로 단어를 밀집 벡터로 만드는 작업을 워드 임베딩(Word Embedding)이라고 한다.
밀집 벡터는 워드 임베딩 과정을 통해 나온 결과므로 임베딩 벡터(Embedding vector)라고도 한다.
원-핫 벡터의 차원이 주로 20,000 이상을 넘어가는 것과는 달리 임베딩 벡터는 주로 256, 512, 1024 등의 차원을 가진다.
임베딩 벡터는 초기에는 랜덤값을 가지지만, 인공 신경망의 가중치가 학습되는 방법과 같은 방식으로 값이 학습되며 변경된다.

- Embedding(): Embedding()은 단어를 밀집 벡터로 만드는 역할을 한다.
  인공 신경망 용어로는 임베딩 층(Embedding Layer)을 만드는 역할을 한다.
  Embedding()은 정수 인코딩이 된 단어들을 입력을 받아서 임베딩을 수행한다.

Embedding()은 (number of samples, input_length)인 2D 정수 텐서를 입력받는다.
이 때 각 sample은 정수 인코딩이 된 결과로, 정수의 시퀀스이다.
Embedding()은 워드 임베딩 작업을 수행하고 (number of samples, input_length, embedding word dimentionality)인 3D 텐서를 리턴한다.

아래의 코드는 실제 동작되는 코드가 아니라 의사 코드(pseudo-code)로 임베딩의 개념 이해를 돕기 위해서 작성되었다.

```python
# 문장 토큰화와 단어 토큰화
text=[['Hope', 'to', 'see', 'you', 'soon'],['Nice', 'to', 'see', 'you', 'again']]

# 각 단어에 대한 정수 인코딩
text=[[0, 1, 2, 3, 4],[5, 1, 2, 3, 6]]

# 위 데이터가 아래의 임베딩 층의 입력이 된다.
Embedding(7, 2, input_length=5)
# 7은 단어의 개수. 즉, 단어 집합(vocabulary)의 크기이다.
# 2는 임베딩한 후의 벡터의 크기이다.
# 5는 각 입력 시퀀스의 길이. 즉, input_length이다.

# 각 정수는 아래의 테이블의 인덱스로 사용되며 Embeddig()은 각 단어에 대해 임베딩 벡터를 리턴한다.
+------------+------------+
|   index    | embedding  |
+------------+------------+
|     0      | [1.2, 3.1] |
|     1      | [0.1, 4.2] |
|     2      | [1.0, 3.1] |
|     3      | [0.3, 2.1] |
|     4      | [2.2, 1.4] |
|     5      | [0.7, 1.7] |
|     6      | [4.1, 2.0] |
+------------+------------+
# 위의 표는 임베딩 벡터가 된 결과를 예로서 정리한 것이고 Embedding()의 출력인 3D 텐서를 보여주는 것이 아님.
```

Embedding()에 넣어야하는 대표적인 인자는 다음과 같다.

- 첫번째 인자 = 단어 집합의 크기. 즉, 총 단어의 개수
- 두번째 인자 = 임베딩 벡터의 출력 차원. 결과로서 나오는 임베딩 벡터의 크기
- input_length = 입력 시퀀스의 길이

---

### 6.3 모델링(Modeling)

- Sequential(): 케라스에서는 입력층, 은닉층, 출력층을 구성하기 위해 Sequential()을 사용한다.
  Sequential()을 model로 선언한 뒤에 model.add()라는 코드를 통해 층을 단계적으로 추가한다.
  아래는 model.add()로 층을 추가하는 예제 코드를 보여준다.
  실제로는 괄호 사이에 있는 온점 대신에 실제 층의 이름을 기재해야 한다.

```python
from tensorflow.keras.models import Sequential
model = Sequential()
model.add(...) # 층 추가
model.add(...) # 층 추가
model.add(...) # 층 추가
```

Embedding()을 통해 생성하는 임베딩 층(embedding layer) 또한 인공 신경망의 층의 하나이므로 model.add()로 추가해야 한다.

```python
from tensorflow.keras.models import Sequential
model = Sequential()
model.add(Embedding(vocabulary, output_dim, input_length))
```

- Dense(): 전결합층(Fully-conntected Layer)을 추가한다.
  model.add()를 통해 추가할 수 있다.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(1, input_dim=3, activation='relu'))
```

위의 코드에서 Dense()는 한번 사용되었지만 더 많은 층을 추가할 수 있다.
Dense()의 대표적인 인자를 봐보자.

- 첫번째 인자 = 출력 뉴런의 수
- input_dim = 입력 뉴런의 수(입력의 차원)
- activation = 활성화 함수
- linear = 디폴트 값으로 별도 활성화 함수 없이 뉴런과 가중치의 계산 결과 그대로 출력
  Ex) 선형회귀
- sigmoid: 시그모이드 함수. 이진 분류 문제에서 출력층에 주로 사용되는 활성화 함수
- softmax: 소프트맥스 함수.  셋 이상을 분류하는 다중 클래스 분류 문제에서 출력층에 주로 사용되는 활성화 함수
- relu: 렐루 함수. 은닉층에 주로 사용되는 활성화 함수

위 코드에서 사용된 Dense()의 의미를 보자.
첫번째 인자의 값은 1인데 이는 총 1개의 출력 뉴런을 의미한다.
Dense()의 두번째 인자인 input_dim은 입력층의 뉴런 수를 의미한다.
이 경우에는 3이다.
3개의 입력층 뉴런과 1개의 출력층 뉴런을 만들었다.
이를 시각화하면 다음과 같다.

![img](https://wikidocs.net/images/page/32105/neural_network1_final.PNG)

이제 Dense()를 사용하여 전결합층을 하나 더 추가해보자.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # 출력층
```

이번에는 Dense()가 두번 사용되었다.
Dense()가 처음 사용되었을 때와 추가로 사용되었을 때의 인자는 조금 다르다.
이제 첫번째 사용된 Dense()의 8이라는 값은 더 이상 출력층의 뉴런이 아니라 은닉층의 뉴런이다.
뒤에 층이 하나 더 생겼기 때문이다.

두번쨔 Dense()는 input_dum 인자가 없는데, 이는 이미 이전층의 뉴런의 수가 8개라는 사실을 알고 있기 때문이다.
위의 코드에서 Dense()는 마지막 층이므로, 첫번째 인자 1은 결국 출력층의 뉴런의 개수가 된다.
이를 시각화하면 다음과 같다.

![img](https://wikidocs.net/images/page/32105/neural_network2_final.PNG)

이 외에도 LSTM, GRU, Convolution2D, BatchNormalization 등 다양한 층을 만들 수 있다.
일부는 나중에 배우게 된다.

- summary(): 모델의 정보를 요약해서 보여준다.

```python
# 위의 코드의 연장선상에 있는 코드임.
model.summary()
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 8)                 40        
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 9         
=================================================================
Total params: 49
Trainable params: 49
Non-trainable params: 0
_________________________________________________________________
```

---

### 6.4 컴파일(Compile)과 훈련(Training)

- compile(): 모델은 기계가 이해할 수 있도록 컴파일한다.
  오차 함수와 최적화 방법, 메트릭 함슈룰 선택할 수 있다.

```python
 이 코드는 뒤의 텍스트 분류 챕터의 스팸 메일 분류하기 실습 코드를 갖고온 것임.
from tensorflow.keras.layers import SimpleRNN, Embedding, Dense
from tensorflow.keras.models import Sequential
max_features = 10000

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32)) #RNN에 대한 설명은 뒤의 챕터에서 합니다.
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
```

위 코드는 임베딩층, 은닉층, 출력층을 추가하여 모델을 설계한 후에, 마지막으로 컴파일 하는 과정을 보여준다. 

- optimizer: 훈련 과정을 설정하는 옵티마이저를 설정한다.
  'adam'이나 'sgd'와 같이 문자열로 지정할 수 있다.
- loss: 훈련 과정에서 사용할 손실 함수(loss function)를 설정한다.
- metrics: 훈련을 모니터링하기 위한 지표를 선택한다.

대표적으로 사용되는 손실 함수와 활성화 함수의 조합은 아래와 같다.
더 많은 함수는 케라스 공식문서에서 확인 가능하다.

|   문제 유형    |                  손실 함수명                   | 출력층의 활성화 함수명 |                          참고 설명                           |
| :------------: | :--------------------------------------------: | :--------------------: | :----------------------------------------------------------: |
|    회귀문제    |       mean_squared_error(평균 제곱 오차)       |           -            |                              -                               |
| 다중클래스분류 | categorical_crossentropy(범주형 교차 엔트로피) |       소프트맥스       |                로이터 뉴스 분류하기 실습 참고                |
| 다중클래스분류 |        sparse_categorical_crossentropy         |       소프트맥스       | 범주형 교차 엔트로피와 동일하지만 이 경우 원-핫 인코딩이 된 상태일 필요없이 정수 인코딩 된 상태에서 수행 가능 |
|    이진분류    |    binary_crossentropy(이항 교차 엔트로피)     |       시그모이드       |    스팸 메일 분류하기, IMDB 리뷰 감성 분류하기 실습 참고     |

- fit(): 모델을 학습한다.
  모델이 오차로부터 매개 변수를 업데이트 시키는 과정을 학습, 훈련 또는 적합(fitting)이라고 하기도 하는데, 모델이 데이터에 적합해가는 과정이기 때문이다.
  그런 의미에서 fit()은 모델의 훈련을 시작한다는 의미를 가지고 있다.

```python
# 위의 compile() 코드의 연장선상인 코드
model.fit(X_train, y_train, epochs=10, batch_size=32)
```

- 첫번째 인자 = 훈련 데이터에 해당
- 두번째 인자 = 지도 학습에서 레이블 데이터에 해당
- epochs = 에포크
  에포크 1은 전체 데이터를 한 차례 훑고 지나갔음을 의미, 정수값 기재 필요, 총 훈련 횟수를 정의
- batch_size = 배치 크기
  기본값은 32, 미니 배치 경사 하강법을 사용하고 싶지 않을 경우에는 batch_size=None을 기재한다.

```python
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, validation_data(X_val, y_val))
```

- validation_data(x_val, y_val) = 검증 데이터(validation data)를 사용한다.
  검증 데이터를 사용하면 각 에포크마다 검증 데이터의 정확도도 함께 출력되는데, 이 정확도는 훈련이 잘 되고 있는지를 보여줄 뿐이며 실제로 모델이 검증 데이터를 학습하지는 않는다.
  검증 데이터의 loss가 낮아지다가 높아지기 시작하면 이는 과적합(overfitting)의 신호이다.
- validation_split = validation_data 대신 사용할 수 있다.
  검증 데이터를 사용하는 것은 동일하지만, 별도로 존재하는 검증 데이터를 주는 것이 아니라 X_train과 y_train에서 일정 비율을 분리하여 이를 검증 데이터로 사용한다.
  역시나 훈련 자체에는 반영되지 않고 훈련 과정을 지켜보기 위한 용도로 사용된다.
  아래는 validation_data 대신에 validation_split을 사용했을 경우를 보여준다.

```python
# 훈련 데이터의 20%를 검증 데이터로 사용.
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, validation_split=0.2)
```

- verbose = 학습 중 출력되는 문구를 설정한다.
  0: 아무 것도 출력하지 않는다.
  1: 훈련의 진행도를 보여주는 진행 막대를 보여준다.
  2: 미니 배치마다 손실 정보를 출력한다.

아래는 verbose의 값이 1일 때와 2일 때를 보여준다.

```python
# verbose = 1일 경우.
Epoch 88/100
7/7 [==============================] - 0s 143us/step - loss: 0.1029 - acc: 1.0000
```

```python
# verbose = 2일 경우.
Epoch 88/100
 - 0s - loss: 0.1475 - acc: 1.0000
```

---

### 6.5 평가(Evaluation)와 예측(Prediction)

- evaluate(): 테스트 데이터를 통해 학습한 모델에 대한 정확도를 평가한다.

```python
# 위의 fit() 코드의 연장선상인 코드
model.evaluate(X_test, y_test, batch_size=32)
```

- 첫번째 인자 = 테스트 데이터에 해당
- 두번째 인자 = 지도 학습에서 레이블 데이터에 해당
- batch_size = 배치 크기
- predict(): 임의의 입력에 대한 모델의 출력값을 확인

```python
# 위의 fit() 코드의 연장선상인 코드
model.predict(X_input, batch_size=32)
```

- 첫번째 인자 = 예측하고자 하는 데이터
- batch_size = 배치 크기

---

### 6.6 모델의 저장(Save)과 로드(Load)

복습응을 위한 스터디나 실제 어플리케이션 개발 단계에서 구현한 모델을 저장하고 불러오는 일은 중요하다.
모델을 저장한다는 것은 학습이 끝난 신경망의 구조를 보존하고 계속해서 사용할 수 있다는 의미이다.

- save(): 인공 신경망 모델을 hdf5 파일에 저장한다.

```python
model.save("model_name.h5")
```

- load_model(): 저장해둔 모델을 불러온다.

```python
from tensorflow.keras.models import load_model
model = load_model("model_name.h5")
```

---

### 6.7 함수형 API(functional API)

대부분의 실습은 위에서 배운 Sequential API를 통해 이루어진다.
위의 코드들은 사용하기에 매우 간단하지만, 복잡한 모델을 설계하기 위해서는 부족함이 있다.

sequential API는 여러층을 공유하거나 다양한 종류의 입력과 출력을 사용하는 등의 복잡한 모델을 만드는 일을 하기에는 한계가 있다.
이번에는 복잡한 모델을 만드는 일을 하기에는 한계가 있다.
이번에는 복잡한 모델을 생성할 수 있는 방식인 functional API(함수형 API)에 대해서 알아보자.

1. sequential API로 만든 모델
   두 가지 API의 차이를 이해하기 위해서 앞서 배운 sequential API를 사용하여 기본적인 모델을 만들어보자.

   ```python
   # 이 코드는 소프트맥스 회귀 챕터에서 가져온 코드임.
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Dense
   model=Sequential()
   model.add(Dense(3, input_dim=4, activation='softmax'))
   ```

   위와 같은 방식은 직관적이고 편리하지만 단순히 층을 쌓는 것만으로는 구현할 수 없는 복잡한 인공 신경망을 구현할 수 없다.

2. functional API로 만든 모델
   functional API는 각 층을 일종의 함수(function)로서 정의한다.
   그리고 각 함수를 조합하기 위한 연산자들을 제공하는데, 이를 이용하여 신경망을 설계한다.
   functional API로 FFNN, RNN 등 다양한 모델을 만들면서 기존의 sequential API와의 차이를 이해해보자.

   1. 전결한 피드 포워드 신경망(Fully-connected FFNN)
      sequential API와는 다르게 functional API에서는 입력 데이터의 크기(shape)를 인자로 입력층을 정의해주어야 한다.
      여기서는 입력의 차원이 1인 전결합 피드 포워드 신경망(Fully-connected FFNN)을 만든다고 가정해보자.

      ```python
      from tensorflow.keras.layers import Input
      # 텐서를 리턴한다.
      inputs = Input(shape=(10,))
      ```

      위의 코드는 10개의 입력을 받는 입력층을 보여준다.
      이제 위의 코드에 은닉층과 출력층을 추가해보자.

      ```python
      from tensorflow.keras.layers import Input, Dense
      inputs = Input(shape=(10,))
      hidden1 = Dense(64, activation='relu')(inputs)
      hidden2 = Dense(64, activation='relu')(hidden1)
      output = Dense(1, activation='sigmoid')(hidden2)
      ```

      이제 위의 코드를 하나의 모델로 구성해보자.
      이는 Model에 입력 텐서와 출력 텐서를 정의하여 완성된다.

      ```python
      from tensorflow.keras.layers import Input, Dense
      from tensorflow.keras.models import Model
      inputs = Input(shape=(10,))
      hidden1 = Dense(64, activation='relu')(inputs)
      hidden2 = Dense(64, activation='relu')(hidden1)
      output = Dense(1, activation='sigmoid')(hidden2)
      model = Model(inputs=inputs, outputs=output)
      ```

      지금까지의 내용을 정리하면 다음과 같다.

      - Input() 함수에 입력의 크기를 정의한다.
      - 이전층을 다음층 함수의 입력으로 사용하고, 변수에 할당한다.
      - Model() 함수에 입력과 출력을 정의한다.

      이제 이를 model로 저장하면 sequential API를 사용할 때와 마찬가지로 model.complie, model.fit 등을 사용 가능하다.

      ```python
      model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
      model.fit(data, labels)
      ```

      이번에는 변수명을 달리해서 FFNN을 만들어보자.
      이번에는 은닉층과 출력층의 변수를 전부 x로 통일했다.

      ```python
      inputs = Input(shape=(10,))
      x = Dense(8, activation="relu")(inputs)
      x = Dense(4, activation="relu")(x)
      x = Dense(1, activation="linear")(x)
      model = Model(inputs, x)
      ```

      이번에는 위에서 배운 내용을 바탕으로 선형 회귀와 로지스틱 회귀를 functional API로 구현해보자.

   2. 선형 회귀(Linear Regression)

      ```python
      from tensorflow.keras.layers import Input, Dense
      from tensorflow.keras.models import Model
      
      inputs = Input(shape=(3,))
      output = Dense(1, activation='linear')(inputs)
      linear_model = Model(inputs, output)
      
      linear_model.compile(optimizer='sgd', loss='mse')
      linear_model.fit(x=dat_test, y=y_cts_test, epochs=50, verbose=0)
      linear_model.fit(x=dat_test, y=y_cts_test, epochs=1, verbose=1)
      ```

   3. 로지스틱 회귀(Logistic Regression)

      ```python
      from tensorflow.keras.layers import Input, Dense
      from tensorflow.keras.models import Model
      
      inputs = Input(shape=(3,))
      output = Dense(1, activation='sigmoid')(inputs)
      logistic_model = Model(inputs, output)
      
      logistic_model.compile(optimizer='sgd', loss = 'binary_crossentropy', metrics=['accuracy'])
      logistic_model.optimizer.lr = 0.001
      logistic_model.fit(x=dat_train, y=y_classifier_train, epochs = 5, validation_data = (dat_test, y_classifier_test))
      ```

   4. 다중 입력을 받는 모델(model that accepts multiple inputs)
      functional API를 사용하면 아래와 같이 다중 입력과 다중 출력을 가지는 모델도 만들 수 있다.

      ```python
      # 최종 완성된 다중 입력, 다중 출력 모델의 예
      model=Model(inputs=[a1, a2], outputs=[b1, b2, b3]
      ```

      이번에는 다중 입력을 받는 모델을 입력층부터 출력층까지 설계해보자.

      ```python
      from tensorflow.keras.layers import Input, Dense, concatenate
      from tensorflow.keras.models import Model
      
      # 두 개의 입력층을 정의
      inputA = Input(shape=(64,))
      inputB = Input(shape=(128,))
      
      # 첫번째 입력층으로부터 분기되어 진행되는 인공 신경망을 정의
      x = Dense(16, activation="relu")(inputA)
      x = Dense(8, activation="relu")(x)
      x = Model(inputs=inputA, outputs=x)
      
      # 두번째 입력층으로부터 분기되어 진행되는 인공 신경망을 정의
      y = Dense(64, activation="relu")(inputB)
      y = Dense(32, activation="relu")(y)
      y = Dense(8, activation="relu")(y)
      y = Model(inputs=inputB, outputs=y)
      
      # 두개의 인공 신경망의 출력을 연결(concatenate)
      result = concatenate([x.output, y.output])
      
      # 연결된 값을 입력으로 받는 밀집층을 추가(Dense layer)
      z = Dense(2, activation="relu")(result)
      # 선형 회귀를 위해 activation=linear를 설정
      z = Dense(1, activation="linear")(z)
      
      # 결과적으로 이 모델은 두 개의 입력층으로부터 분기되어 진행된 후 마지막에는 하나의 출력을 예측하는 모델이 됨.
      model = Model(inputs=[x.input, y.input], outputs=z)
      ```

   5. RNN(Recurrence Neural Network) 은닉층 사용하기
      이번에는 RNN 은닉층을 가지는 모델을 설계해보자.
      여기서는 하나의 특상(feature)에 50개의 시점(time-step)을 입력으로 받는 모델을 설계해보자.

      ```python
      from tensorflow.keras.layers import Input, Dense, LSTM
      from tensorflow.keras.models import Model
      inputs = Input(shape=(50,1))
      lstm_layer = LSTM(10)(inputs) # RNN의 일종인 LSTM을 사용
      x = Dense(10, activation='relu')(lstm_layer)
      output = Dense(1, activation='sigmoid')(x)
      model = Model(inputs=inputs, outputs=output)
      ```

   6. 다르게 보이지만 동일한 표기
      케라스의 functional API가 익숙하지 않은 상태에서 functional API를 사용한 코드를 보다가 혼동할 수 있는 점이 한 가지 있다.
      바로 동일한 의미를 가지지만, 하나의 줄로 표현할 수 있는 코드를 두 개의 줄로 표현한 경우이다.

      ```python
      encoder = Dense(128)(input)
      ```

      위 코드는 아래와 같이 두 개의 줄로 표현할 수 있다.

      ```python
      encoder = Dense(128)
      encoder(input)
      ```

      

