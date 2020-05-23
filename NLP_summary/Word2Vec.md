# Word2Vec


Source: https://wikidocs.net/22660
Word2Vec은 단어 간 유사도를 반영할 수 있도록, 단어의 의미를 벡터화 할 수 있는 방법이다.

---

### 1. 희소 표현(Sparse Representation)

One-hot encoding을 통해서 나온 One-hot Vector들은 표현하고자 하는 단어의 인덱스의 값만 1이고, 나머지 인덱스에는 전부 0으로 표현되는 방법이다.
이렇게 vector 또는 matrix의 값이 대부분 0으로 표현되는 방법을 희소 표현이라고 한다.

하지만 이러한 표현 방법은 각 단어간 유사성을 표현할 수 없다는 문제점이 존재한다.
그렇기에 이를 위한 대안으로 단어의 '의미'를 다차원 공간에 벡터화하는 방법을 구현하는데, 이 표현 방법을 분산 표현(distributed representation)이라고 한다.
그리고 이렇게 분산 표현을 이용하여 단어의 유사도를 벡터화하는 작업은 임베딩(embedding) 작업에 속하며, 이 작업을 통해 표현된 vector는 embedding vector라고 한다.

---

### 2. 분산 표현(Distributed Representation)

Distributed Representation 방법은 기본적으로 분포 가설(distributional hypothesis)이라는 가정 하에 만들어진 표현 방법이다.
이 가정은 '비슷한 위치에서 등장하는 단어들은 비슷한 의미를 가진다'라는 내용을 가진다.
예시로 강아지라는 단어는 귀엽다, 예쁘다, 애교 등의 단어가 주로 함께 등장하는 걸, 분포 가설에 따라서 저런 내용을 가진 텍스트를 벡터화한다면 단어들은 의미적으로 가까운 단어가 된다.

이렇게 표현된 vector들은 one-hot vector처럼 벡터의 차원이 단어 집합의 크기가 되지 않으므로, 벡터의 차원이 상대적으로 저차원으로 줄어들게 된다.
Ex1) 강아지 = [ 0 0 0 0 1 0 0 0 0 0 0 0 ... 중략 ... 0] 
Ex2) 강아지 = [1, 2]

---

### 3. CBOW(Continuous Bag of Words)

Word2Vec에는 CBOW와 Skip-Gram, 두 가지 방식이 존재한다.
CBOW는 한 단어를 기준으로, 주변에 있는 단어를 이용하여 기준점으로 하는 단어를 예측하는 방법이다.
그 와 반대로, Skip-Gram은 기준점이 되는 단어로 주변에 있는 단어를 예측하는 방법이다.
두 방법은 목적이 반대이지만, 매커니즘 자체는 매우 유사하다.

예문: "The fat cat sat on the mat"
다음과 같은 예문이 있다고 가정하고, {"The", "fat", "cat", "on", "the", "mat"}에서 sat을 예측하는 것은 CBOW의 방법이다.
이 때 예측해야하는 단어 sat을 중심 단어(center word)라고 하고, 예측에 사용되는 단어들을(context word)라고 한다.

중심 단어를 예측하기 위해서 앞, 뒤로 몇 개의 단어를 확인할지를 결정했다면 이 범위를 window라고 한다.
예시로 window가 2라면 기준이 되는 단어를 기준점으로 앞, 뒤로 2개의 단어를 확인하게 되는 것이다. 

![img](https://wikidocs.net/images/page/22660/%EB%8B%A8%EC%96%B4.PNG)

윈도우 크기를 정한 후, 윈도우를 계속 움직이며 주변 단어와 중심 단어 선택을 바꿔가며 학습을 위한 데이터 셋을 만들 수 있다.
이 방법을 슬라이딩 윈도우(sliding window)라고 한다.

위 그림처럼 좌측의 중심 단어와 주변 단어의 변화는 윈도우 크기가 2일 때, 슬라이딩 윈도우가 어떤 식으로 이루어지면서 데이터 셋을 만드는지 보여준다.

![img](https://wikidocs.net/images/page/22660/word2vec_renew_1.PNG)

CBOW의 인공 신경망을 간단히 도식화하면 위 그림과 같다.
Input Layer의 입력으로 사용자가 정한 윈도우 크기 범위 안에 있는 주변 단어들의 one-hot vector가 들어가게 되고, Output Layer에서 예측하고자 하는 중간 단어의 one-hot vector가 필요하다.

또한, Word2Vec은 딥러닝 모델이 아니다.
보통 딥러닝은 Input Layer와 Output Layer 사이의 Hidden Layer의 개수가 충분히 쌓여 있는 신경망을 지칭한다.
그러나 Word2Vec은 Input Layer와 Output Layer 사이에 하나의 Hidden Layer만이 존재한다.
이렇게 Hidden Layer가 1개인 경우, 일반적으로 DNN(Deep Neural Network)이 아닌 SNN(Shallow Neural Network)이라고 한다.
또한 Word2Vec의 Hidden Layer은 일반적으로 은닉층과 달리 활성화 함수가 존재하지 않으며, 룩업 테이블이라는 연산을 담당하는 층으로 일반적인 Hidden Layer와 구분하기 위해 투사층(Projection Layer)이라고 부르기도 한다.

![img](https://wikidocs.net/images/page/22660/word2vec_renew_2.PNG)

CBOW의 인공 신경망을 좀 더 확대하여, 동작 매커니즘에 대해서 더 상세하게 알아볼 수 있다.
위 그림에서 주목해야하는 것은 두 가지이다.
하나는 투사층의 크기가 M이라는 점이다.
CBOW애서 투사층의 크기 M은 임베딩하고 난 벡터의 차원이 됩니다.
다시 말해, 위의 그림에서 투사층의 크기는 M=5이기 때문에 CBOW를 수행하고 나서 얻는 각 단어의 임베딩 벡터의 차원은 5가 될 것입니다.

두번째는 Input Layer와 Output Layer 사이의 가중치 W는 V * M 행렬이며, 투사층에서 Output Layer 사이의 가중치 W'은 M * V 행렬이라는 점이다.
여기서 V는 단어 집합의 크기를 의미한다.
즉, 위의 그림처럼 one-hot vector의 차원이 7이고, M은 5라면 가중치 W는 7 * 5 행렬이고, W'는 5 * 7 행렬이 될 것이다.
주의할 점은 이 두 행렬은 동일한 행렬을 전치(Transpose)한 것이 아닌, 서로 다른 행렬이하는 점이다.
ANN의 훈련 전에는 이 가중치 행렬 W와 W'은 대게 굉장히 작은 랜덤 값을 가지게 된다.
CBOW는 주변 단어로 중심 단어를 더 정확하게 맞추기 위해 계속해서 이 W와 W'를 학습해가는 구조이다.

![img](https://wikidocs.net/images/page/22660/word2vec_renew_3.PNG)

입력으로 들어오는 주변 단어의 one-hot vector와 가중치 W 행렬의 곱이 어떻게 이루어지는지 볼 수 있다.
위 그림에서 각 주변 단어의 one-hot vector를 ![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Math/Italic/400/0078.png?V=2.7.1)로 표기했다.
입력 벡터는 one-hot vector이다.
i번째 인덱스에 1이라는 값을 가지고 그 외의 0의 값을 가지는 입력 벡터와 가중치 W 행렬의 곱은 사실 W행렬의 i번째 행을 그대로 읽어오는 것과 동일(lookup)하다.
그래서 이 작업을 룩업 테이블(Lookup Table)이라고 지칭한다.
앞서 CBOW의 목적은 W와 W'를 잘 훈련시키는 것이라고 언급한 적이 있는데, 사실 그 이유가 여기서 lookup 해온 W의 각 행벡터가 사실 Word2Vec을 수행한 후 의 각 단어의 M차원의 크기를 갖는 임베딩 벡터들이기 때문이다.

![img](https://wikidocs.net/images/page/22660/word2vec_renew_4.PNG)

이렇게 각 주변 단어의 one-hot vector에 대해서 가중치 W가 곱해서 생겨진 결과 벡터들은 투사층에서 만나 이 벡터들의 평균인 벡터를 구하게 된다.
만약 윈도우 크기가 2라면, 입력 벡터의 총 개수는 2n이므로 중간 던어를 예측하기 위해서는 총 4개가 입력 벡터로 사용된다.
그렇기 때문에 평균을 구할 때는 4개의 결과 벡터에 대해서 평균을 구하게 된다.
투사층에서 벡터의 평균을 구하는 부분은 CBOW가 Skip-Gram과 다른 차이점이기도 한다.
나중에 나오는 Skip-Gram은 입력이 중심 단어 하나이기 때문에 투사층에서 벡터의 평균을 구하지 않는다.

![img](https://wikidocs.net/images/page/22660/word2vec_renew_5.PNG)

이렇게 구해진 평균 벡터는 두번째 가중치 행렬 W'와 곱해진다.
곱셈의 결과로는 one-hot vector들과 차원이 V로 동일한 벡터가 나온다.
만약 입력 벡터의 차원이 7이었다면, 여기서 나오는 벡터 또한 마찬가지다.

이 벡터에 CBOW는 softmax함수를 취하는데, softmax 함수로 인한 출력값은 0과 1 사이의 실수로, 각 원소의 총 합은 1이 되는 상태로 바뀐다.
이렇게 나온 벡터를 스코어 벡터(score vector)라고 한다.
스코어 벡터의 각 차원 안에서의 값이 의미하는 것은 다음과 같다.

스코어 벡터의 j번째 인덱스가 가진 0과 1사이의 값은 j번째 단어가 중심 단어일 확률을 나타낸다.
그리고 이 스코어 벡터는 우리가 실제로 값을 알고 있는 벡터인 중심 단어 one-hot vector의 값에 가까워져야 한다.
스코어 벡터를 ![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Math/Italic/400/0079.png?V=2.7.1)^라고 가정하여, 중심단어를  ![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Math/Italic/400/0079.png?V=2.7.1)로 했을때, 이 두 벡터값의 오차를 줄이기 위해 CBOW는 손실 함수(Loss Function)로 cross-entropy 함수를 사용한다. 

![img](https://wikidocs.net/images/page/22660/crossentrophy.PNG)

cross-entropy 함수에 실제 중심 단어인 one-hot vector와 스코어 벡터를 입력값으로 넣고, 이를 식으로 표현하면 위 그림과 같다.

![img](https://wikidocs.net/images/page/22660/crossentrophy2.PNG)

그런데 y가 one-hot vector라는 점을 고려하면, 이 식은 위와 같이 간소화 시킬 수 있다.
이 식이 왜 loss function에 적합한지 설명할 수 있다.
c를 중심 단어에서 1을 가진 차원의 값의 인덱스라고 한다면, ![img](https://wikidocs.net/images/page/22660/best.PNG)는 ![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Math/Italic/400/0079.png?V=2.7.1)^가 ![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Math/Italic/400/0079.png?V=2.7.1)를 정확하게 예측한 경우가 된다.
이를 식에 대입해 보면, -1 log(1)=0이 되기 때문에, 결과적으로 ![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Math/Italic/400/0079.png?V=2.7.1)^가 ![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Math/Italic/400/0079.png?V=2.7.1)를 정확하게 예측한 경우의 cross-entropy의 값은 0이 된다.
즉, ![img](https://wikidocs.net/images/page/22660/crossentrophy.PNG)이 값을 최소화는 방향으로 학습을 진행해야 한다.

이제 역전파(Back Propagation)를 수행하면 W와 W'가 학습이 되는데, 학습이 다 되었다면 M차원의 크기를 갖는 W의 행이나 W'의 열로부터 어떤 것을 임베딩 벡터로 사용할지를 결정하면 된다.
때로는 W와 W'의 평균치를 가지고 임베딩 벡터를 선택하기도 한다.

---

### 4. Skip-Gram

Skip-Gram과 CBOW는 메커니즘 자체는 동일하기 때문에 쉽게 이해할 수 있다.
CBOW에서는 주변 단어를 통해 중심 단어를 예측했다면, Skip-Gram은 중심 단어에서 주변 단어를 예측한다.

![img](https://wikidocs.net/images/page/22660/word2vec_renew_6.PNG)

앞서 언급한 동일한 예문에 대해서 ANN을 도식화해보면 위 그림과 같다.
이제 중심 단어에 대해서 주변 단어를 예측하기에, 투사층에서 벡터들의 평균을 구하는 과정은 없다.

여러 논문에서 성능 비교를 진행했을 때, 전반적으로 Skip-Gram이 CBOW보다 성능이 좋다고 알려져 있다.

---

### 5. NNLM vs Word2Vec

![img](https://wikidocs.net/images/page/22660/word2vec_renew_7.PNG)

NNLM은 단어 간 유사도를 구할 수 있도록 워드 임베딩의 개념을 도입하고, NNLM의 느린 학습 속도와 정확도를 개선하여 탄생한 것이 Word2Vec이다.

NNLM과 Word2Vec의 차이를 비교해보자면, 우선 예측하는 대상이 다르다.
NNLM은 언어 모델이기에 다음 단어를 예측하지만, Word2Vec(CBOW)은 워드 임베딩 자체가 목적이므로 다음 단어가 아닌 중심 단어를 예측하여 학습한다.
중심단어를 예측하게 하므로서 NNLM이 예측 단어의 이전 단어들만을 참고하였던 것과는 달리, Word2Vec은 예측단어의 전, 후 단어들을 모두 참고한다.

다음으로는 구조가 달라졌다.
위 그림은 n을 학습에 사용하는 단어의 수, m을 임베딩 벡터의 차원, h를 Hidden Layer의 크기, V를 단어 집합의 크기라고 했을 때 NNLM과 Word2Vec의 차이를 보여준다.
Word2Vec은 NNLM에 존재하던 활성화 함수가 있는 Hidden Layer을 제거했다.
이에 따라 투사층 다음에 바로 Output Layer으로 연결되는 구조다.

Word2Vec이 NNLM보다 학습 속도에서 강점을 가지는 이유는 은닉층을 제거한 것뿐만 아니라 추가적으로 사용되는 기법들 덕분이기도 하다.
대표적으로 계층적 소프트맥스(hierarchical softmax)와 네거티브 샘플링(negative sampling)이 있는데 여기서는 네거티브 샘플링만 언급한다.
Word2Vec과 NNLM의 연산량을 비교하여 학습 속도가 왜 차이가 나는지 이해할 수 있다.

우선 Input Layer에서 투사층에서 은닉층, 은닉층에서 Output Layer으로 향하며 발생하는 NNLM의 연상량을 보면,
NNLM:  ![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Main/Regular/400/0028.png?V=2.7.1)![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Math/Italic/400/006E.png?V=2.7.1)![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Main/Regular/400/00D7.png?V=2.7.1)![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Math/Italic/400/006D.png?V=2.7.1)![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Main/Regular/400/0029.png?V=2.7.1)![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Main/Regular/400/002B.png?V=2.7.1)![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Main/Regular/400/0028.png?V=2.7.1)![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Math/Italic/400/006E.png?V=2.7.1)![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Main/Regular/400/00D7.png?V=2.7.1)![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Math/Italic/400/006D.png?V=2.7.1)![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Main/Regular/400/00D7.png?V=2.7.1)![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Math/Italic/400/0068.png?V=2.7.1)![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Main/Regular/400/0029.png?V=2.7.1)![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Main/Regular/400/002B.png?V=2.7.1)![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Main/Regular/400/0028.png?V=2.7.1)![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Math/Italic/400/0068.png?V=2.7.1)![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Main/Regular/400/00D7.png?V=2.7.1)![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Math/Italic/400/0056.png?V=2.7.1)![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Main/Regular/400/0029.png?V=2.7.1)

추가적인 기법들까지 사용했을 때, Word2Vec은 Output Layer에서의 연산에서 ![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Math/Italic/400/0056.png?V=2.7.1)를 ![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Math/Italic/400/006C.png?V=2.7.1)![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Math/Italic/400/006F.png?V=2.7.1)![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Math/Italic/400/0067.png?V=2.7.1)![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Main/Regular/400/0028.png?V=2.7.1)![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Math/Italic/400/0056.png?V=2.7.1)![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Main/Regular/400/0029.png?V=2.7.1)로 바꿀 수 있는데, 이에 따라 Word2Vec의 연산량은 아래와 같으며 이는 NNLM보다는 배는 빠른 학습 속도를 가진다.
Word2Vec:  ![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Main/Regular/400/0028.png?V=2.7.1)![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Math/Italic/400/006E.png?V=2.7.1)![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Main/Regular/400/00D7.png?V=2.7.1)![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Math/Italic/400/006D.png?V=2.7.1)![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Main/Regular/400/0029.png?V=2.7.1)![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Main/Regular/400/002B.png?V=2.7.1)![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Main/Regular/400/0028.png?V=2.7.1)![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Math/Italic/400/006D.png?V=2.7.1)![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Main/Regular/400/00D7.png?V=2.7.1)![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Math/Italic/400/006C.png?V=2.7.1)![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Math/Italic/400/006F.png?V=2.7.1)![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Math/Italic/400/0067.png?V=2.7.1)![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Main/Regular/400/0028.png?V=2.7.1)![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Math/Italic/400/0056.png?V=2.7.1)![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Main/Regular/400/0029.png?V=2.7.1)![img](https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/fonts/HTML-CSS/TeX/png/Main/Regular/400/0029.png?V=2.7.1)

---

### 6. Negative Sampling

대체적으로 Word2Vec을 사용한다고 하면, SGNS(Skip-Gram with Negative Sampling)을 사용한다.
Skip-Gram을 사용하는데, Negative Sampling이란 방법까지 추가적으로 사용하는 거다.

위에서 배운 Word2Vec 모델에는 한 가지 문제점이 있다.
그 문제는 속도이다.
Word2Vec의 마지막 단계를 봐보면, Output Layer에 있는 softmax function는 단어 집합 크기의 벡터 내의 모든 값을 0과 1 사이의 값이면서 모두 더하면 1이 되도록 바꾸는 작업을 수행한다.
그리고 이에 대한 오차를 구하고 모든 단어에 대한 임베딩을 조정한다.
그 단어가 중심 단어나 주변 단어와 전혀 상관없는 단어이더라도 마찬가지다.
그런데 만약 단어 집합의 크기가 수백만에 달한다면 이 작업은 굉장히 무거운 작업이다.

여기서 중요한 건 Word2Vec이 모든 단어 집합에 대해서 softmax function을 수행하고, 역전파를 수행하므로 주변 단어와 상관 없는 모든 단어까지의 워드 임제딩 조정 작업을 수행한다는 것이다.
예시로, 마지막 단계에서 '강아지'와 '고양이'와 같은 단어에 집중하고 있다면, Word2Vec은 사실 '돈가스'나 '컴퓨터'와 같은 연관 관계가 없는 수 많은 단어의 임베딩을 조정할 필요가 없다.

이를 조금 더 효율적으로 할 수 있는 방법이 없을지, 전체 단어 집합이 아니라 일부 단어 집합에 대해서만 고려하면 안되는지에 대한 의문이 생긴다.
이렇게 일부 단어 집합을 만들어 보자.
'강아지', '고양이', '애교'와 같은 주변 단어들을 가져온다.
그리고 여기에 '돈가스', '컴퓨터', '회의실'과 같은 랜덤으로 선택된 주변 단어가 아닌 상관없는 단어들을 일부만 가져온다.
이렇게 전체 단어 집합보다 훨씬 작은 단어 집합을 만들어놓고 마지막 단계를 이진 분류 문제로 바꿔버리는 거다.
즉, Word2Vec은 주변 단어들을 긍정으로 두고 랜덤으로 샘플링된 단어들을 부정으로 둔 다음에 이진 분류 문제를 수행한다.

이는 기존의 다중 클래스 분류 문제를 이진 분류 문제로 바꾸면서도 연산량에 있어서 훨씬 효율적이다.