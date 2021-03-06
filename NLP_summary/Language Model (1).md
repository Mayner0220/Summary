# Language Model (1)

Source: https://wikidocs.net/21695, https://wikidocs.net/21668, https://wikidocs.net/21687, https://wikidocs.net/21692, https://wikidocs.net/22533, https://wikidocs.net/21697, https://wikidocs.net/21681

---

### 0. Language Model

언어 모델(Language Model)이란 단어 시퀀스(문장)에 확률을 할당하는 모델을 말한다.
어떤 문장들이 있을 때, 기계가 이 문장을 사람처럼 판단할 수 있다면, 기계가 NLP를 잘 한다고 볼 수 있다.
이게 바로 언어 모델이 하는 일이다.

이번에는 통계에 기반한 전통적인 언어 모델(Statistical Language Model, SLM)에 대해서 배운다.
통계에 기반한 언어 모델은 우리가 실제 사용하는 자연어를 근사하기에는 많은 한계가 있고, 요즘 들어 인공 신경망이 그러한 한계를 많이 해결해주면서 통계 기반 언어 모델은 많이 사용 용도가 줄었다.

하지만 그럼에도 여전히 통계 기반 언어 모델에서 배우게 될 n-gram은 NLP분야에서 활발하게 활용되고 있으며, 통계 기반 방법론에 대한 이해는 언어 모델에 대한 전체적인 시야를 갖는 일에 도움이 된다.

---

### 1. 언어 모델이란?

언어 모델(Language Model, LM)은 언어라는 현상을 모델링하고자 단어 시퀀스(또는 문장)에 확률을 할당(assign)하는 모델이다.

언어 모델을 만드는 방법은 크게는 통계를 이용한 방법과 인공 신경망을 이용한 방법으로 구분할 수 있다.
최근에는 통계를 이용한 방법보다는 인공 신경망을 이용한 방법이 더 좋은 성능을 보여준다.
최근 핫한 NLP의 신기술인 GPT나 BERT 또한 인공 신경망 언어 모델의 개념을 사용하여 만들어졌다.

---

### 1.1 언어 모델(Language Model)

언어 모델은 단어 시퀀스에 확률을 할당(assign)하는 일을 하는 모델이다.
이를 풀어서 쓰면, 언어 모델은 가장 자연스러운 단어 시퀀스를 찾아내는 모델이다.
단어 시퀀스에 확률을 할당하게 하기 위해서 가장 보편적으로 사용되는 방법은 언어 모델이 이전 단어들이 주어졌을 때 다음 단어를 예측하도록 하는 것이다.

다른 유형의 언어 모델로는 주어진 양쪽의 단어들로부터 가운데 비어있는 단어를 예측하는 언어 모델이 있다.

언어 모델에 -ing를 붙인 언어 모델링(Language Modeling)은 주어진 단어들로부터 아직 모르는 단어를 예측하는 작업을 말한다.
즉, 언어 모델이 이전 단어들로부터 다음 단어를 예측하는 일은 언어 모델링이다.

NLP로 유명한 스탠포드 대학교에서는 언어 모델을 문법이라고 비유하기도 한다.
언어 모델이 단어들의 조합이 얼마나 적절한지, 또는 해당 문장이 얼마나 적합한지를 알려주는 일을 하는 것이 마치 문법이 하는 일 같기 때문이다.

---

### 1.2 단어 시퀀스의 확률 할당

NLP에서 단어 시퀀스에 확률을 할당하는 일이 왜 필요할까?
예를 들어보자면, 여기서 대문자 P는 확률을 의미한다.

1. 기계 번역(Machine Translation):
   $$
   P(나는 버스를 탔다) > P(나는 버스를 태운다) 
   $$
   : 언어 모델은 두 문장을 비교하여 좌측의 문장의 확률이 더 높다고 판단한다.

2. 오타 교정(Spell Correction):
   선생님이 교실로 부리나케
   $$
   P(달려갔다) > P(잘려갔다)
   $$
   : 언어 모델은 두 문장을 비교하여 좌측의 문장의 확률이 더 높다고 판단한다.

3. 음성 인식(Speech Recognition):
   $$
   P(나는 메롱을 먹는다) < P(나는 메론을 먹는다)
   $$
   : 언어 모델은 두 문장을 비교하여 우측의 문장의 확률이 더 높다고 판단한다.

언어 모델은 위와 같이 확률을 통해 보다 적절한 문장을 판단한다.

---

### 1.3 주어진 이전 단어들로부터 다음 단어 예측하기

언어 모델은 단어 시퀀스레 확률을 할당하는 모델이다.
그리고 단어 시퀀스에 확률을 할당하기 위해서 가장 보편적으로 사용하는 방법은 이전 단어들이 주어졌을 때, 다음 단어를 예측하도록 하는 것이다.
이를 조건부 확률로 표현해보자.

1. 단어 시퀀스의 확률
   하나의 단어를 w, 단어 시퀀스을 대문자 W라고 한다면, n개의 단어가 등장하는 단어 시퀀스 W의 확률은 다음과 같다.
   $$
   P(W)=P(w1,w2,w3,w4,w5,...,wn)
   $$

2. 다음 단어 등장 확률
   이제 다음 단어 확률을 식으로 표현해보자.
   n-1개의 단어가 나열된 상태에서 n번째 단어의 확률은 다음과 같다.
   $$
   P(wn|w1,...,wn−1)
   $$
   |의 기호는 조건부 확률(conditional probability)을 의미한다.
   예를 들어 다섯번째 단어의 확률은 아래와 같다.
   $$
   P(w5|w1,w2,w3,w4)
   $$
   전체 단어 시퀀스 W의 확률은 모든 단어가 예측되고 나서야 알 수 있으므로, 단어 시퀀스의 확률은 다음과 같다.
   $$
   P(W)=P(w1,w2,w3,w4,w5,...wn)=∏i=1nP(wn|w1,...,wn−1)
   $$

---

### 1.4 언어 모델의 간단한 직관

예문으로, '비행기를 타려고 공항에 갔는데 지각을 하는 바람에 비행기를'이라는 문장이 있다.
'비행기를' 다음에 어떤 단어가 오게 될지 사람은 쉽게 '놓쳤다'라고 예상할 수 있다.
우리 지식에 기반하여 나올 수 있는 여러 단어들을 후보에 놓고 놓쳤다는 단어가 나올 확률이 가장 높다고 판단하였기 때문이다.

그렇다면 기계에서 위 문장을 주고, '비행기를' 다음에 나올 단어를 예측해보라고 한다면 과연 어떻게 최대한 정확히 예측할 수 있을까?
기계도 비슷하다.
앞에 어떤 단어들이 나왔는지 고려하여, 후보가 될 수 있는 여러 단어들에 대해서 확률을 예측해보고 가장 높은 확률을 가진 단어를 선택한다.
앞에 어떤 단어들이 나왔는지 고려하여 후보가 될 수 있는 여러 단어들에 대해서 등장 확률을 추정하고 가장 높은 확률을 가진 간어를 선택한다.

---

### 1.5 검색 엔진에서의 언어 모델의 예

![img](https://wikidocs.net/images/page/21668/%EB%94%A5_%EB%9F%AC%EB%8B%9D%EC%9D%84_%EC%9D%B4%EC%9A%A9%ED%95%9C.PNG)

검색 엔진이 입력된 단어들의 나열에 대해서 다음 단어를 예측하는 언어 모델을 사용하고 있다.

---

### 2. 통계적 언어 모델(Statistical Language Model, SLM)

여기서는 언어 모델의 전통적인 접근 방법인 통계적 언어 모델을 알아본다.
통계적 언어 모델이 통계적 접근 방법으로 어떻게 언어를 모델링하는지 배워보자.

---

### 2.1 조건부 확률

조건부 확률은 P(A), P(B)에 대해서 아래와 같은 관계를 갖는다.
$$
p(B|A)=P(A,B)/P(A)
$$

$$
P(A,B)=P(A)P(B|A)
$$

더 많은 확률에 대해서 일반화해보자.
4개의 확률이 조건부 확률의 관계를 가질 때, 아래 와 같이 표현할 수 있다.
$$
P(A,B,C,D)=P(A)P(B|A)P(C|A,B)P(D|A,B,C)
$$
이를 조건부 확률의 연쇄 법칙(chain rule)이라고 한다.
이제 4개가 아닌 n개에 대해서 일반화 해보자.
$$
P(x1,x2,x3...xn)=P(x1)P(x2|x1)P(x3|x1,x2)...P(xn|x1...xn−1)
$$
조건부 확률에 대한 정의를 통해 문장의 확률을 구해보자.

---

### 2.2 문장에 대한 확률

문장 'An adorable little boy is spreading smiles'의 확률:
$$
P(An adorable little boy is spreading smiles)
$$
를 식으로 해보자.

각 단어는 문맥이라는 관계로 인해 이전 단어의 영향을 받아 나온 단어이다.
그리고 모든 단어로부터 하나의 문자이 완성된다.
그렇기 때문에 문장의 확률을 구하고자 조건부 확률을 사용해보자.
앞서 언급한 조건부 확률의 일반화 식을 문장의 확률 관점에서 다시 적어보면 문장의 확률은 각 단어들이 이전 단어가 주어졌을 때, 다음 단어로 등장할 확률의 곱으로 구성된다.
$$
P(w1,w2,w3,w4,w5,...wn)=∏n=1nP(wn|w1,...,wn−1)
$$
위의 문장에 해당 식을 적용해보면 다음과 같다.
$$
P(An adorable little boy is spreading smiles)=
P(An)×P(adorable|An)×P(little|An adorable)×P(boy|An adorable little)×P(is|An adorable little boy) ×P(spreading|An adorable little boy is)×P(smiles|An adorable little boy is spreading)
$$
문장의 확률을 구하기 위해서 각 단어에 대한 예측 확률들을 곱한다.

---

### 2.3 카운트 기반 접근

문장의 확률을 구하기 위해서 다음 단어에 대한 예측 확률을 모두 곱한다는 것은 알았다.
그렇다면 SLM은 이전 단어로부터, 다음 단어에 대한 확률은 어떻게 구할까.
정답은 카운트에 기반하여 확률을 계산한다.

An adorable little boy가 나왔을 때, is가 나올 확률인 
$$
P(is|An adorable little boy) 
$$
 을 구해보자.
$$
P\text{(is|An adorable little boy}) = \frac{\text{count(An adorable little boy is})}{\text{count(An adorable little boy })}
$$
그 확률은 위와 같다.
예를 들어 기계가 학습한 코퍼스에 데이터에서 An adorable little boy가 100번 등장했는데 그 다음에 is가 등장한 경우는 30번이라고 하자.
이 경우 
$$
P(is|An adorable little boy)
$$
는 30%이다.

---

### 2.4 카운트 기반 접근의 한계 - 희소 문제(Sparsity Problem)

언어 모델은 실생활에서 사용되는 언어의 확률 분포는 근사 모델링 한다.
실제로 정확하게 알아볼 방법은 없겠지만 현실에서도 An adorable little boy가 나왔을 때, is가 나올 확률이라는 것이 존재한다.
이를 실제 자연어의 확률 분포, 현실에서의 확률 분포라고 명칭하자.
기계에서 많은 코퍼스를 훈련시켜서 언어 모델을 통해 현실에서의 확률 분포를 근사하는 것이 언어 모델의 목표이다.
그런데 카운트 기반으로 접근하려고 한다면, 갖고있는 코퍼스(훈련용 데이터)는 정말 방대한 양이 필요하다.
$$
P\text{(is|An adorable little boy}) = \frac{\text{count(An adorable little boy is})}{\text{count(An adorable little boy })}
$$
예를 들어 위와 같이 P(is|An adorable little boy)P(is|An adorable little boy)를 구하는 경우에서 기계가 훈련한 코퍼스에 An adorable little boy is라는 단어 시퀀스가 없었다면 이 단어 시퀀스에 대한 확률은 0이 된다.
또는 An adorable little boy라는 단어 시퀀스가 없었다면 분모가 0이 되어 확률은 정의되지 않는다.
그렇다면 코퍼스에 단어 시퀀스가 없다고 해서 이 확률을 0 또는 정의되지 않는 확률이라고 하는 것이 정확한 모델링 방법일까?
아니다. 현실에선 An adorable little boy is 라는 단어 시퀀스가 존재하고 또 문법에도 적합하므로 정답일 가능성 또한 높다. 
이와 같이 충분한 데이터를 관측하지 못하여 언어를 정확히 모델링하지 못하는 문제를 희소 문제(sparsity problem)라고 한다.

위 문제를 완화하는 방법으로 다음에서 배우는 n-gram이나 스무딩이나 백오프와 같은 여러가지 일반화(generalization) 기법이 존재한다.
하지만 희소 문제에 대한 근본적인 해결책이 되지는 못했다.
결국 이러한 한계로, 언어 모댈의 트렌드는 통계적 언어 모델에서 인공 신경망 언어 모델로 넘어가게 된다.

