# Class Imbalanced

Reference

- [클래스 불균형, UnderSampling & OverSampling](https://hwiyong.tistory.com/266)
- [SMOTE로 데이터 불균형 해결하기](https://medium.com/@john_analyst/smote%EB%A1%9C-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EB%B6%88%EA%B7%A0%ED%98%95-%ED%95%B4%EA%B2%B0%ED%95%98%EA%B8%B0-5ab674ef0b32)

---

### 클래스 불균형(Class Imbalanced)

본론을 들어가기 전, oversampling을 이용하게 되는 상황인 클래스 불균형(Class Imbalanced)에 대해 먼저 알아보자.

클래스 불균형은 이름 그대로 데이터에서 클래스가 불균형하게 분포되어 있다는 걸 나타낸다.

이러한 문제들을 비정상 탐지(Anomaly Detection)이라고 부른다.

---

### Undersampling

과소표집(Undersampling)은 다른 클래스에 비해 상대적으로 많이 존재하는 클래스의 개수를 줄이는 것이다.

이를 통해 균형을 유지할 수 있지만, 제거하는 과정 중에 유용한 정보들이 버려지게 되는 것이 큰 단점이다.

---

### Oversampling

과대표집(Oversampling)은 데이터를 복제하는 것이다.

그 방식으로는 무작위로 하는 경우도 있고, 기준을 미리 정해두고 복제하는 방법도 있다.

정보를 잃지 않고, 훈련용(train) 데이터에서 높은 성능을 보이지만 실험용(validation or test) 데이터에서는 성능이 낮아질 수 있다.

대부분의 과대표집의 방법들은 과적합(Overfitting)의 문제를 가지고 있다.

이를 피하기 위해 주로 SMOTE(Synthetic Minority Over-sampling Technique)를 사용한다.

간단하게 설명하자면, 데이터의 개수가 적은 클래스의 표본(Sample)을 가져온 뒤에 임의의 값을 추가하여 새로운 샘플을 만들어 데이터에 추가한다.

이 과정에서 주변 데이터를 고려하기 때문에 과대적합의 가능성이 낮아지게 된다.

---

### Smote

Smote는 Synthetic minority oversampling technique의 약자이다.

데이터를 수집하고 관리 등에 있어서 데이터의 형태나 상태는 이상적이지 않다.

그 경우의 수 중 하나가, 클래스 불균형이다.

데이터에서 각 클랫의 개수가 현저하게 차이가 난 상태로 모델을 학습하면, 다수의 범주로 패넡 분류를 많이하게 되는 문제가 발생한다.

즉, 모델의 성능에 영향을 끼치게 되는 것이다.

SMOTE의 동작 방식은 데이터의 개수가 적은 클래스의 표본을 가져온 뒤 임의의 값을 추가하여 새로운 샘플을 만들어 데이터에 추가하는 오버샘플링 방식이다.

![Image for post](https://miro.medium.com/max/1380/0*DfTZFQO5nhdiYmiY.png)



