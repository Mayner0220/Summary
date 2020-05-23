# **What's the NLP? (2) :thinking:** 

이 글은 스탠포드 대학에서 제공하는 딥러닝을 이용한 NLP -CS224n- 강의노트(Lecture Note) 1장의 뒷 부분을 번역한 내용입니다.
또한 이 글은 "솔라리스의 인공지능 연구실"의 글을 참조하여 작성되었습니다.
Source: http://solarisailab.com/archives/959

Reference
[1] http://web.stanford.edu/class/cs224n/syllabus.html

[2] http://web.stanford.edu/class/cs224n/lecture_notes/cs224n-2017-notes1.pdf

[3] https://arxiv.org/pdf/1301.3781.pdf

---

아래의 이미지는 **CBOW**와 **Skip-gram**의 차이를 직관적으로 나타낸다.

![img](http://solarisailab.com/wp-content/uploads/2017/05/cbow_and_skip-gram_diff.png)

---

**4. 반복에 기반한 방법 - Word2vec (Iteration based Methods - Word2vec)**
이제 한발짝 물러서서, 새로운 방법을 시도해보자.
많은 양의 전체 데이터셋-많은 양의 문장들을 계산하고 저장하는 대신에, 한번씩 반복을 통헤서 학습하고, 결국에는 주어진 컨텍스트(context)안에서 단어의 확률을 부호화(encode)할 수 있는 모델을 시도할 수 있다.

이 아이디어는 word vector들이 파라미터 모델을 디자인 하는 것이다.
그 다음에, 어떤 목적함수를 가지고 모델을 학습시킨다.
매 반복(iteration)마다 우리의 모델을 실행(run)하고, 오차(errors)를 평가하고, 에러에 대해 페널티(penalizing)를 주는 개념의 업데이트 룰(update rule)을 통해서 모델을 개선한다.
그럼으로써, 우리는 word vector들을 학습한다.
이 아이디어는 매우 오래되었고, 그 근원은 1986년까지 거슬러 올라간다.
이 방법은 오류 역전파(errors backpropagating)라고 불린다.
모델과 문제(task)가 간단할수록, 더욱 빠른 시간안에 그것을 학습할 수 있을 것이다.

몇몇 방법들이 테스트되어 왔다.
논문은 NLP를 위한 첫번째 스텝으로 각각의 단어를 vector로 변환하는 모델을 제안했다.
몇몇 특별한 문제들(Names Entity Recognition, Part-of-Speech tagging, etc...)에 대해서 그들은 모델의 파라미터들뿐만 아니라 벡터들까지 학습했다.
그리고 좋은 word vector들을 계산할 뿐만 아니라 좋은 성능을 보여주었다.

이 수업에서는, 논문에서 제안된 더 간단하고, 더 최신의 확률적 모델인 Word2vec을 다룰 것이다.
Word2vec은 다음것들을 포함한 소프트웨어 패키지이다.

- 2개의 알고리즘: continuous bag-of-words(CBOW) 그리고 skip-gram.
  CBOW는 word vectors를 이용하여 주어진 컨텍스트 상의 중앙에 위치한 단어를 예측하는 것을 목표로 한다.
  반대로, Skip-gram은 중앙 단어로부터 컨텍스트 단어들의 분포(probability)를 예측하려 한다.
- 2개의 학습(training) 방법: negative sampling, 그리고 hierarchical softmax.
  Negative sampling은 목적함수를 negative 예제들을 통해서 샘플링한다.
  반면에, hierarchical softmax는 모든 단어들에 대한 확률을 계산하기 위한 효율적인 트리 구조(tree structure)를 이용해서 목적함수를 정의한다.

**4.1 Language Models (Unigrams, Bigrams, etc...)**
먼저, 우리는 각각의 토큰(token)들에 확률을 부여하기 위한 모델을 만들 필요가 있다.
다음 예제와 함께 시작해보자.

**“The cat jumped over the puddle”**

좋은 language model은 이 문장에 높은 확률을 부여할 것이다.
왜냐하면, 이 문장은 의미적으로나(semantically), 구조적으로나(syntactically) 완전히 유의미한(valid) 문장이기 때문이다.
유사하게, "stock boil fish is toy"라는 문장은 전혀 말이 안되는 문장이기 때문에 낮은 확률을 부여해야한다.
수학적으로, ![n](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-e6e1cf6806f410e6c96555565b366f6c_l3.png)개의 단어로 이루어진 어떤 문장이라도 이런 확률을 부여할 수 있다.

![\begin{equation*} P(w_1,w_2,\cdots,w_n) \end{equation*}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-7179f57a569d01dcb71290d686f7d22e_l3.png)

우리는 단항의(unary) language model 방법을 취하고 등장이 완전히 독립적(independent)이란 가정하에 이런 확률들을 분해할 수 있다.

![\begin{equation*} P(w_1,w_2,\cdots,w_n)=\prod_{i=1}^n{P(w_i)} \end{equation*}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-ab7f759e1f68b96bda778cdd736d79a7_l3.png)

하지만, 우리는 이런 방법이 터무니 없다는 사실을 이미 알고 있다.
왜냐하면 우리는 이전의 단어들의 묶음과 이후에 나올 단어가 매우 큰 연관관계를 가지고 있다는 사실을 알고 있기 때문이다.
그리고 이 방법을 이용하면 엉터리 문장도 높은 점수를 얻을 수 있다.
따라서 우리는 문장의 확률이 문장의 묶음과 이후에 나오는 단어 쌍의 확률에 의존하도록 하고 싶다.
이런 방법은 bigram model이라고 부르고 다음과 같이 나타낼 것이다.

![\begin{equation*} P(w_1,w_2,\cdots,w_n)=\prod_{i=2}^n{P(w_i|w_{i-1})} \end{equation*}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-46949fef3360effed9c713d7f3aec19b_l3.png)

다시, 이 방법은 전체 문장을 고려하는 것이 아니라, 오직 이웃한 단어쌍만을 고려하기 때문에 나이브(naive)한 방법이라는 것을 알 수 있다.
하지만 볼 수 있듯이, 이런 표현은 우리가 모델링을 어느 정도 진행했음을 알려준다.
컨텍스트 크기(context of size)를 1로한 Word-Word Matrix를 이용해서, 이런 단어쌍의 확률을 기초적으로 학습할 수 있다.
하지만 이런 방법은 또 다시 거대한 전체 데이터셋에 대한 전역 정보(global information)를 계산하고 저장하는 과정이 필요하다.

이제 토큰들을 확률로 나타내는 방법을 이해했으니, 이런 확률들을 학습할 수 있는 예제 모델들을 알아보자.

**4.2 Continuous Bag of Words Models (CBOW)**
하나의 방법은 {"The", "cat", "over", "the", "puddle"}을 컨텍스트로 취급하고, 이런 단어들로 부터 중앙에 단어인 "jumped"를 예측하는 것이다.
이런 종류의 모델을 우리는 Continuous Bag of Words (CBOW) Model이라고 부를 것이다.

이제 위에 언급한 CBOW 모델을 자세하게 알아보자.
먼저, 우리가 알고있는 파라미터들을 설정해야만 한다.
모델에서 우리가 알고 있는 파라미터들은 one-hot vector들로 나타내어진 문장이 될 것이다.
input으로 주어진 one-hot vector들과 context는 ![x^{(c)}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-7cdc83e25cd0587829339ea521e922b9_l3.png)로 나타내어 질 것이다.
그리고 CBOW 모델에서의 output은 ![y^{(c)}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-f44902fa0848fd295283d1691c203527_l3.png)로 나타내어질 것이다.
우리는 오직 하나의 output만을 가지고 있기 때문에, 그냥 ![y](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-5e6848524627bf4505a639de0146cefc_l3.png)라고 표기할 것이다.
이는 알고 있는 중앙 단어(center word)를 one-hot vector로 표현한 것이다.
이제 우리 모델에서 알고 있지 못한 것들을 정의해보자.

우리는 두 개의 행렬들을 만들 수 있다.
![V \in R^{n\times|V|}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-22dc3ed63c26df27e78c4d70140dee34_l3.png)그리고 ![U \in R^{|V|\times{n}}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-9b44888994d351f1553c3f0da016a9be_l3.png).
![n](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-e6e1cf6806f410e6c96555565b366f6c_l3.png)은 우리의 embadding space의 크기를 정의하는 임의의 크기를 가진 변수이다.
![V](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-06140836e79dbd6b6b00b48fa51e0167_l3.png)는 ![V](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-06140836e79dbd6b6b00b48fa51e0167_l3.png)의 ![i](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-cc64060f364aab1f0e662c8a8a0816c4_l3.png)번째 column이 input 단어 ![w_i](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-127cb061a8014bcfacd5b32b43402581_l3.png)를 표현한 ![n](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-e6e1cf6806f410e6c96555565b366f6c_l3.png)차원의 embedded vector인 input word matrix이다.
우리는 이 ![n\times1](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-022a961d24bcedf83cc2f5d85e056fcb_l3.png)vector를 ![v_i](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-777786c7ba6fc6b0db4b7b2e0da64543_l3.png)로 표기할 것이다.
유사하게, ![U](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-02cfda02c2fd09347e3f499c66782b47_l3.png)는 output matrix이다.
![U](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-02cfda02c2fd09347e3f499c66782b47_l3.png)의 ![j](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-f565589cb98b3eaf1dfd6ad0a48aafcd_l3.png)번째 row는 output단어 ![w_j](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-222e3cd881b76410c73354bada3d3db5_l3.png)의 ![n](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-e6e1cf6806f410e6c96555565b366f6c_l3.png)차원의 embedded vector이다.
우리는 ![u](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-cdded677d5d320ef0a5c30651528e5a3_l3.png)의 이런 row를 ![u_{ij}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-b8c003bf04d46ac5ccb5dca846c5d699_l3.png)로 표기할 것이다.
우리가 모든 단어 ![w_i](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-127cb061a8014bcfacd5b32b43402581_l3.png)에 대해서, 두개의 vector들을 학습한다는 사실을 기억하라.

이제 모델이 하는 일을 단계별로 나누어서 생각해보자.

1. 크기 ![m](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-659547cfd4ed7c77d59397cb6c59bdf1_l3.png)의 input 컨텍스트(context)를 위한 one-hot vectors를 생성한다:
   ![(x^{(c-m)},...,x^{(c-1)},x^{(c+1)},...,x^{(c+m)} \in R^{|V|})](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-03b4758882efdcd15f1aeb0388eaf25b_l3.png)
2. context ![(v_{c-m} = Vx^{(c+m)},v_{c-m+1}=Vx^{(c-m+1)},...,v_{c+m}=Vx^{(c+m)} \in R^n)](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-57fd919d211e5418e73a9002f9bd3530_l3.png)를 위한 embedded word vectors를 얻는다.
3. ![\hat{v}=\frac{v_{c-m}+v_{c-m+1}+...+v_{c+m}}{2m} \in R^n](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-99ef11d9cd2d45265528311d545538b2_l3.png)를 얻디 위해 벡터들의 평균을 취한다.
4. score vector ![z=U\hat{v} \in R^{|V|}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-0e86cc91463b50dacf45ef0fa4b007a9_l3.png)를 만든다. 비슷한 벡터들의 내적(dot product)은 높은 값을 갖고, 이는 높은 점수를 얻기 위헤서 비슷한 단어들끼리 가까이 위치하도록 강제한다.
5. score들을 확률![\hat{y}=softmax(z) \in R^{|V|}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-40dcfb6706d825d4bff28d8f25f300ff_l3.png)로 바꾼다.
6. 우리는 우리가 만든 확률 ![\hat{y} \in R^{|V|}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-00fceb1ea160b1d4b8efb63aadd4f536_l3.png)가 실제 확률 ![y \in R^{|V|}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-f57b00b05ecccfd419b833dc0f027f0a_l3.png)와 일치하고, 실제 단어의 one-hot vector와 일치하기를 원한다.

이제 우리는 ![V](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-06140836e79dbd6b6b00b48fa51e0167_l3.png)와 ![U](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-02cfda02c2fd09347e3f499c66782b47_l3.png)를 가지고 있을 때, 우리의 모델이 어떻게 작동하는지 이해했다.
그렇다면, 이 두 개의 행렬 ![U](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-02cfda02c2fd09347e3f499c66782b47_l3.png)와 ![V](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-06140836e79dbd6b6b00b48fa51e0167_l3.png)는 어떻게 학습할까?
이를 위해서 목적함수(objective function)를 정의해야만 한다.
true probability로 부터 probability를 학습하려고 할때 자주 사용하는 방법은, 정보 이론(information theory)을 이용한 두 개의 분포(distributions)간의 거리를 측정하는 방법이다.
여기서, 우리는 대중적인 distance/loss measure인 cross entropy ![H(\hat{y},y)](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-eade59ab00137a05626c586f5d461140_l3.png)를 사용할 것이다.

discrete case에서 cross-entropy를 사용하는 직관(intuition)은 다음의 손실 함수(loss function)을 정의함으로써 얻을 수 있다.

![\begin{equation*} H(\hat{y},y)=-\sum_{j=1}^{|V|}y_{j}log(\hat{y}_j) \end{equation*}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-cf2e4df35db44371e9a8f5015d627f54_l3.png)

우리가 다루는 문제의 경우 ![y](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-5e6848524627bf4505a639de0146cefc_l3.png)가 one-hot vector라고 가정하였다.
따라서 우리는 위의 loss를 함수를 아래와 같이 좀 더 간단한 형태로 바꿀 수 있다.

![\begin{equation*} H(\hat{y},y)=-y_{ㅑ}log(\hat{y}_ㅑ) \end{equation*}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-f32d306af3103567afa86088c85bbb80_l3.png)

우리가 다루는 문제의 경우 ![y](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-5e6848524627bf4505a639de0146cefc_l3.png)가 one-hot vector라고 가정하였다.
따라서 우리는 위의 loss 함수를 아래와 같이 좀 더 간단한 형태로 바꿀 수 있다.

![\begin{equation*} H(\hat{y},y)=-y_{ㅑ}log(\hat{y}_ㅑ) \end{equation*}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-f32d306af3103567afa86088c85bbb80_l3.png)

이 정의에서, ![c](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-325e2c76ae9b9226bb22391a49daf11f_l3.png)는 정확한 단어(correct word)의 one-hot vector가 1인 비율이 얼마인지 나타내는 index이다.
이제 우리의 예측이 완벽해서 ![\hat{y}_c=1](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-ffcf3d3c7ae5077c0171c461690c9856_l3.png)인 상황을 가정하자.
그러면 ![H(\hat{y},y)=-1log(1)=0](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-692be93a16daf0926c8f815c70971373_l3.png)임을 알 수 있다.
따라서, 완벽한 예측을 위해서, 우리는 loss를 0으로 만들어야 한다.
이제 반대의 경우로, 우리의 예측이 매우 잘못되어서 ![\hat{y}_c= 0.01](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-04a7e75ae7e6330cc5a74026bed39c27_l3.png)인 경우를 가정해보자.
이 경우 loss를 계산 해보면 ![H(\hat{y},y)=-1log(0.01)\approx 4.605](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-fd6ea32b0245dc97ca423609eb601df4_l3.png)이다.
따라서 우리는 probability distribution에 대해서 cross entopy가 좋은 distance measure로 사용될 수 있다는 점을 알 수 있다.
따라서 목적함수에 대한 우리의 최적화(optimization)를 다음과 같이 수식화 해보자.

![\begin{equation*} \begin{split} minimize\ J&= -logP(w_c|w_{c-m},...,w_{c-1},w_{c+1},...,w_{c+m}) \\ &= -logP(u_{c}|\hat{v}) \\ &=-log\frac{exp(u_c^T\hat{v})}{\sum_{j=1}^{|V|}exp(u_j^T\hat{v})} \\ &=-u^T_c\hat{v}+log\sum_{j=1}^{|V|}exp(u_j^T\hat{v}) \end{split} \end{equation*}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-b02e6bd2f629d83a5c17d36b9d7d63db_l3.png)

우리는 모든 연관된 word vectors와 ![u_c](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-baaa4dd51b779d350db9033931890d62_l3.png)와 ![v_j](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-3b00856919cf604a92baabb09920a69f_l3.png)를 업데이트하기 위해서 stochastic gradient descent를 사용할 수 있다.

**4.3 Skip-Gram Model**

모델을 생성하기 위한 또 다른 방법은 주어진 중앙 단어 "jumped"를 이용해서 주변의 연관된 단어 "The", "cat", "over", "the", "puddle"을 예측하는 것이다.
여기서 우리는 단어 "jumped"를 컨텍스트(context)로 부를 것이다.
이런 종류의 모델을 Skip-Gram model이라고 부른다.

이제 Skip-Gram 모델을 알아보자.
구성은 CBOW와 거의 유사하다.
단지 x와 y를 바꾸는 것이 차이이다.
CBOW에서 x는 y가 되고 y는 x가 된다.
input one hot vector(중앙 단어)는 오직 하나이기 때문에 ![x](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-2c2455128ee4a55994f0c51012317375_l3.png)라고 표현한다.
output vectors는 ![y^{(j)}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-e36a31aafcaef42d1f6e4a0cdca6939b_l3.png)로 표기한다.
CBOW 경우와 마찬가지로 ![V](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-06140836e79dbd6b6b00b48fa51e0167_l3.png)와 ![U](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-02cfda02c2fd09347e3f499c66782b47_l3.png)를 정의한다.
이제 모델이 하는 일을 단계별로 나누어서 생각해보자.

1. 중앙 단어(center word)를 이용하여 one hot input vector ![x \in R^{|V|}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-d1a908d1812d04824fff02d2f7238770_l3.png)를 생성한다.
2. 중앙 단어 ![v_c=Vx \in R^n](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-72e0bdb71bbe70cb7597caed7ac788ca_l3.png)를 위한 embedded word vector를 구한다.
3. score vector ![z=Uv_c](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-d0dc30c50feb63224fb37206e4ac1383_l3.png)를 생성한다.
4. score vector를 ![\hat{y}=softmax(z)](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-e363f7199e3ebf3a67429ad22cfa4fcc_l3.png)확률로 변환한다.
   ![\hat{y}_{c-m},...,\hat{y}_{c-1},\hat{y}_{c+1},...,\hat{y}_{c+m}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-fa3c17da59d6997ba07935ac287f3d3a_l3.png)는 각각의 중앙 단어를 관측하면서 생성된 probabilities이다.
5. 우리는 우리가 만든 probability vector가 실제 확률 ![y^{(c-m)},...,y^{(c-1)},y^{(c+1)},...y^{(c+m)}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-af1fd07a46e433301e280665b15fb958_l3.png)와 일치하고, 실제 단어의 one-hot vector와 일치하기를 원한다.

CBOW에서와 마찬가지로, 우리의 모델을 평가하기 위해서, 목적함수(objective function)를 생성하는 것이 필요하다.
주요한 차이는 우리가 Naive Bayes assumption을 이용해서 확률을 계산한다는 점이다.
만약 당신이 이것에 대해 던에 들어본 적이 없다면, 간단히 말해서, 이는 강력한 조건부 독립(conditional indenpendence) 가정이라고 생각하면 된다. 다르게 표현하면, 주어진 중앙 단어에 대해서 모든 output 단어들은 완전히 독립적이다.

![\begin{equation*} \begin{split} minimize\ J&= -logP(w_{c-m},...,w_{c-1},w_{c+1},...,w_{c+m}|w_c) \\ &=-log\prod_{j=0,j\neq{m}}^{2m}P(w_{c-m+j}|w_c) \\ &=-log\prod_{j=0,j\neq{m}}^{2m}P(u_{c-m+j}|v_c) \\ &=-log\prod_{j=0,j\neq{m}}^{2m}\frac{exp(u^T_{c-m+j}v_c)}{\sum_{k=1}^{|V|}exp(u_k^Tv_c)} \\ &=-\sum_{j=0,j\neq{m}}^{2m}u^T_{c-mj+j}v_c+2mlog\sum_{k=1}^{|V|}exp(u_k^Tv_c) \\ \end{split} \end{equation*}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-edb68ff12278aab4d40cbe2799f0af45_l3.png)

이 목적 함수와 함께, 우리는 각 interation마다 unknown 파라미터들에 대한 gradients를 계산할 수 있고, 각 interation마다 Stochastic Gradient Descent를 이용하여 파라미터들을 업데이트 할 수 있다.
수식으로 표현하면,

![\begin{equation*} \begin{split} J&=-\sum_{j=0,j\neq{m}}^{2m}logP(u_{c-m+j}|v_c)\\ &=\sum_{j=0,j\neq{m}}^{2m}H(\hat{y},y_{c-m+j})\\ \end{split} \end{equation*}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-97f8eb3e6de4a6926428d0d2ec059ce3_l3.png)

![H(\hat{y},y_{c-m+j})](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-78540255e80b6aea3cbc1d99db12e7a6_l3.png)는 probability vector ![\hat{y}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-833e9b04bb2d3bd1e9b56433c4f62a08_l3.png)와 one-hot vector ![y_{c-m+j}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-b591dc91b02702d7102dae9ab1468656_l3.png)간의 cross-entropy이다.

**4.4 Negative sampling**

objective function을 들여다보기 위한 단계로 넘어가 보자.
![|V|](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-c40bbcc0a59a786cf47412de36ed7c9f_l3.png)합을 계산하는 것은 계산량이 매우 많이 필요하다는 사실을 기억하자.
objective function을 업데이트하거나 평가하기 위해서는 ![O(|V|)](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-de45d25adb3d5fc4733ab1324eac99e0_l3.png)시간이 걸린다.
이를 해결하기 위한 간단한 아이디어는 직접 계산하기보다는 단지 이를 근사하는(approximate) 방법이다.

매 training step마다 전체 단어들을 훑어보는 대신에, 우리는 단지 몇몇의 부정적인 예저들(negative examples)을 샘플링 할 수 있다.
우리는 단어들의 빈도를 정렬한 noise distribution ![(P_n(w))](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-7cce6ea98339e980a6746b255cb2e8e2_l3.png)으로 부터 "sample"할 수 있다.
우리의 방법을 Negative Sampling과 결합하기 위해서 단지 필요한 일은, 다음의 것들을 업데이트하는 것이다.

- objective function
- gradients
- update rules

Mikolov et al.은 "Distributed Representation of Words and Phrases and their Compositionality"라는 논문에서 Negative Sampling을 제안했다. Negative Sampling이 Skip-Gram 모델에 기반하고 있지만 사실 이는 다른 목적함수를 최적화한다.
![(w,c)](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-704809714746a14f1f44bff4cce2b9ff_l3.png)의 단어와 컨텍스트 쌍을 고려해보자.
이 쌍이 트레이닝 데이터로부터 왔는가?
![(w,c)](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-704809714746a14f1f44bff4cce2b9ff_l3.png) 쌍이 corpus 데이터로부터 추출되었을 확률을 ![P(D=1|w,c)](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-7e7c9c9460704ae81ab535c0934b2cc2_l3.png)으로 표기하자.
동시에, ![(w,c)](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-704809714746a14f1f44bff4cce2b9ff_l3.png)쌍이 corpus 데이터로부터 추출되지 않았을 확률을 ![P(D=0|w,c)](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-eb4a29a51d0fdcc21fccaac70bc2fe62_l3.png)으로 표기하자.
첫번째로, ![P(D=1|w,c)](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-7e7c9c9460704ae81ab535c0934b2cc2_l3.png)를 시그모이드(sigmoid function)를 이용해서 모델링해보자.

![\begin{equation*} P(D=1|w,c,\theta)=\sigma(v_c^Tv_w)=\frac{1}{1+e^{(-v_c^Tv_w)}} \end{equation*}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-40c8352e070429ffd89b43342ec87d83_l3.png)

이제 우리는 word와 context가 실제로 corpus data안에 있다면 corpus data에 있을 확률을 최대화하고, word와 context가 실제로 corpus data안에 없다면 corpus data에 없을 확률을 최대화하는 새로운 목적 함수를 만들어보자.
우리는 이 두 확률에 대해 간단한 maximum likelihood 방법을 취할 수 있다. (![\theta](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-b0b477354bb9711ebe24ffdac9f60745_l3.png)을 모델의 파라미터로 실정할 것이다. 이번 문제의 경우 이는 ![V](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-06140836e79dbd6b6b00b48fa51e0167_l3.png)와 ![U](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-02cfda02c2fd09347e3f499c66782b47_l3.png)를 나타낸다.)

![\begin{equation*} \begin{split} \theta&= \underset{\theta}{\arg\max}\prod_{(w,c) \in D}P(D=1|w,c,\theta) \prod_{(w,c) \in \tilde{D}}P(D=0 |w,c,\theta) \\ &=\underset{\theta}{\arg\max}\prod_{(w,c) \in D}P(D=1|w,c,\theta) \prod_{(w,c) \in \tilde{D}}(1-P(D=1 |w,c,\theta)) \\ &=\underset{\theta}{\arg\max}\sum_{(w,c) \in D}logP(D=1|w,c,\theta)+\sum_{(w,c) \in \tilde{D}}log(1-P(D=1 |w,c,\theta)) \\ &=\underset{\theta}{\arg\max}\sum_{(w,c) \in D}log\frac{1}{1+exp(-u^T_wv_C)}+\sum_{(w,c) \in \tilde{D}}log(1-\frac{1}{1+exp(-u^T_wv_c)}) \\ &=\underset{\theta}{\arg\max}\sum_{(w,c) \in D}log\frac{1}{1+exp(-u^T_wv_C)}+\sum_{(w,c) \in \tilde{D}}log(\frac{1}{1+exp(u^T_wv_c)}) \\ \end{split} \end{equation*}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-5035f52eee3b111ea20d59c6b5460067_l3.png)

likelihood를 최대화(maximizing)하는 것은 negative log likelihood를 최소화(minimizing)하는 것과 같다.

![\begin{equation*} J =-\sum_{(w,c) \in D}log\frac{1}{1+exp(-u^T_wv_c)}-\sum_{(w,c)\in \tilde{D}}log(\frac{1}{1+exp(u_w^Tv_c)}) \end{equation*}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-68d2cd84012d2a5863120f09cb9656a1_l3.png)

![\tilde{D}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-7cccc70eabf0ac25cb390f1457484d51_l3.png)는 "false" 또는 "negative" corpus이다.
"stock boil fish is toy" 문장과 같은 부자연스러운 문장들은 낮은 확률을 가질 것이다.
우리는 word bank에서 임의로 샘플링해가면서 ![\tilde{D}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-7cccc70eabf0ac25cb390f1457484d51_l3.png)를 생성할 수 있다.

Skip-gram 모델에서, 주어진 중앙 단어 ![c](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-325e2c76ae9b9226bb22391a49daf11f_l3.png)로부터 context 단어 ![c - m + j](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-23b36e7d20e2aec2208b66f5faedb9b3_l3.png)를 관찰하는 우리의 새로운 목적함수는 다음과 같다.

![\begin{equation*} -log\sigma(u^T_{c-m+j}\cdot{v_c})-\sum_{k=1}^Klog\simga(-{\tilde{u}_{k}^{T}} \cdot v_c) \end{equation*}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-5d164ee6f5a208f65644e8c9b40f41a9_l3.png)

CBOW 모델에서, 주어진 context vector ![\hat{v}=\frac{v_{c-m}+v_{c-m+1}+...+v_{c+m}}{2m}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-9a0e5c766f501789460a7a1fbe178ea3_l3.png)로부터 중앙 단어 ![u_c](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-baaa4dd51b779d350db9033931890d62_l3.png)를 관찰하는 우리의 새로운 목적함수는 다음과 같다.

![\begin{equation*} -log\sigma(u^T_{c}\cdot\hat{v})-\sum_{k=1}^{K}log\sigma(-\tilde{u}_k^{T}\cdot\hat{v}) \end{equation*}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-37285727b9917d0eeaee6ac8f1861351_l3.png)

위의 수식에서, ![\{\tilde{u}_k|k=1...K\}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-57d04561360860451aeea01c487f2e84_l3.png)는 ![P_n(w)](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-7ddb48fadf77f24d5e1a442654dff796_l3.png)로부터 샘플링되었다.
이제 ![P_n(w)](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-7ddb48fadf77f24d5e1a442654dff796_l3.png)이 무엇인지 논의해보자.
무엇이 가장 좋은 근사(approximation)인지에 대한 많은 논의가 있지만, 가장 좋아 보이는 방법은 3/4 제곱을 이용한 Unigram Model이다.
왜 3/4인가?
이에 대한 직관적 이해를 돕기 위한 예제는 아래와 같다.

- is: ![0.9^{3/4}=0.92](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-0705843dfdab4b5833289a9f12c3fad5_l3.png)
- Constitution: ![0.09^{3/4}=0.16](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-aa19f04b9b2aff81e0696604c2ab468f_l3.png)
- bombastic: ![0.01^{3/4} = 0.032](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-f58170831b5e98f7ab24f05b52f4057d_l3.png)

이제, "is"가 가끔 샘플링 되는 반면에 "bombastic"은 3배 이상 자주 샘플링 될 것이다.

**4.5 Hierarchical softmax**

또한 Mikolov et al.는 일반적인(normal) softmax보다 훨씬 효율적인 hierarchical softmax를 제안했다.
실제 상황에서, hierarchical softmax는 자주 등장하지 않는 단어에 대해 더 잘 작동하고, negative sampling은 자주 등장하는 단어와 저차원(lower dimensional) 벡터들에 대해 더 잘 작동하는 경향이 있다.

Hierarchical softmax는 어휘 안에 모든 단어들을 나타내기 위해 이진트리(binary tree)를 사용한다.
각각의 트리에서 각각의 leaf는 단어를 나타낸다.
그리고 root로부터 leaf까지 유일한 경로(unique path)를 가지고 있다.
이 모델에서는 단어들을 위한 output 표현이 없다.
대신에, 그레프 상의 각각의 노드(root와 leaves를 제외한)들은 모델이 학습하고 있는 vector와 연관을 맺고 있다.

이 모델에서, 주어진 vector ![w_i](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-127cb061a8014bcfacd5b32b43402581_l3.png)에서 단어 ![w](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-d99b40b4d568f33c34174286e1539cc2_l3.png)의 확률 ![P(w|w_i)](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-32e3f50865fbe4b68f4edb6fea8a3573_l3.png)는 root로부터 w에 대응되는 leaf node의 끝까지의 random walk 확률을 구하는 것과 같다.
이런 식으로 확률을 계산함으로써 얻는 이점은, 계산향이 오직 경로의 길이에 대응되는 ![O(log|V|)](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-a92f86efd0738c994be215dcdeb5a39e_l3.png)밖에 안된다는 점이다.

![img](http://solarisailab.com/wp-content/uploads/2017/05/binary_tree_for_hierarchical_softmax.png)

- 그림 1 - Hierarchical softmax를 위한 이진트리(Binary tree)

몇몇 표기들을 소개하겠다.
![L(w)](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-3f4bf37f080fd7d10d8d4af6bee147a7_l3.png)는 root로부터 leaf ![w](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-d99b40b4d568f33c34174286e1539cc2_l3.png)까지의 경로에 있는 노드들의 숫자이다.
예를 들어, 그림 1의 ![L(w_2)](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-15b5491aec6441ca152986c240236951_l3.png)는 3이다.
![n(w,i)](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-af18f9be3f6163ca94ebb03220d9f957_l3.png)는 vector ![v_{n(w,i)}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-a42c73388c685f2afec94bf27d497c04_l3.png)와 연관된 경로의 ![i](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-cc64060f364aab1f0e662c8a8a0816c4_l3.png)번째 노드이다.
따라서, ![n(w,1)](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-bfc78cd8eb12103dbc0a777040cbb1be_l3.png)은 root이고, ![n(w,L(w))](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-d9d8da061cb7816eed69f7d8945698f3_l3.png)는 ![w](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-d99b40b4d568f33c34174286e1539cc2_l3.png)의 부모노드이다.
이제, 각각의 inner node ![n](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-e6e1cf6806f410e6c96555565b366f6c_l3.png)에서 임의로 이의 자식노드를 고르고 이를 ![ch(n)](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-8b3308a6f9c7b334ec75b593964d7051_l3.png)이라고 부르자.
그 다음에, 우리는 다음과 같이 확률을 계산할 수 있다.

![\begin{equation*} P(w|w_i) = \prod_{j=1}^{L(w)-1}\sigma([n(w,j+1)=ch(n(w,j))]\cdot v^T_{n(w,j)}v_{w_i}) \\ \end{equation*}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-034486e306b036c312d9ec21187a1d84_l3.png)

![where](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-099aad8c51d85a1cf895731709d334eb_l3.png)

![\begin{equation} \[ [x]=\left\{ \begin{array}{ll} 1 \ if \ x \ is \ true \\ -1 \ otherwise\\ \end{array} \right. \] \end{equation}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-cfb3e8137da746fd7a12e2026a4b4e2c_l3.png)

![\sigma(\cdot)](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-fd29eb6261f78c2d85866c18a9f0684b_l3.png)은 시그모이드 함수(sigmoid function)이다.

이 수식은 매우 압축되어 표현되어 있다.
따라서 좀 더 자세하게 설명해보자.

먼저, root ![(n(w,1))](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-8dc859782547c7b58a8849154fc5ff68_l3.png)로부터 leaf ![(w)](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-0ef80c90aaf518bdff80d67abc1eac36_l3.png)까지의 경로의 모양에 기반해서 내적을 계산한다.
만약 우리가 ![ch(n)](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-8b3308a6f9c7b334ec75b593964d7051_l3.png)이 항상 ![n](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-e6e1cf6806f410e6c96555565b366f6c_l3.png)의 왼쪽 노드라고 가정한다면, term ![[n(w,j+1)=ch(n(w,j))]](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-39f7f0f077d32fe3708dfddb8a9c64d6_l3.png)는 경로가 왼쪽으로 간다면 1, 오른쪽으로 간다면 -1을 return할 것이다.

더욱이, term ![[n(w,j+1)=ch(n(w,j))]](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-39f7f0f077d32fe3708dfddb8a9c64d6_l3.png)는 정규화(normalization)를 제공한다.
노드 ![n](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-e6e1cf6806f410e6c96555565b366f6c_l3.png)에서 우리가 왼쪽 노드로 갈 확률과 오른쪽 노드로 갈 확률을 더한다면, 어떤 값의 ![v^T_{n}v_{wi}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-26ab7faffb913c0cbab08a7a7195c33d_l3.png)도 확인할 수 있다.

![\begin{equation*} \sigma(v_n^Tv_{w_i})+\sigma(-v_n^Tv_{w_i}) = 1 \end{equation*}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-86807705303db4b756079d2bc1b5a49a_l3.png)

이 정규화는 또한 original softmax처럼 ![\sum_{w=1}^{|V|}P(w|w_i) = 1](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-6e89f72c46c82ab603b8418ceec1e051_l3.png)임을 보장한다.

마지막으로, 우리는 input vector ![w_i](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-127cb061a8014bcfacd5b32b43402581_l3.png)와 inner node vector v^T_{n(w,j)}간의 유사도(similarity)를 내적(dot product)을 통해서 구할 수 있다.
예제를 통해서 살펴보자.
그림 1에서 ![w_2](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-e2219b5a59458196445b1b9489385fe8_l3.png)를 취하고, root로부터 ![w_2](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-e2219b5a59458196445b1b9489385fe8_l3.png)까지 도달하기 위해서 두 개의 왼쪽 모서리(edges)들을 취하고 하나의 오른쪽 모서리를 취한다.
그러면, 

![\begin{equation*} \begin{split} P(w_2|w_i)&= p(n(w_2,1),left)\cdot p(n(w_2,2),left)\cdot p(n(w_2,3),right) \\ &=\sigma(v^T_{n(w_2,1}v_{w_i})\cdot\sigma(v^T_{n(w_2,2)v_{w_i}})\cdot\sigma(-v^T_{n(w_2,3)}v_{w_i}) \\ \end{split} \end{equation*}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-9c81146b9d782de8aabb9739b2f2b608_l3.png)

모델을 학습하기 위해서, 우리의 목적은 여전히 negative log likelihood ![-logP(w|w_i)](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-69b81125900811025296318e1725d203_l3.png)를 최소화하는 것이다.
하지만 각각의 단어들에 대한 output vector를 업데이트하는 대신에, root로부터 leaf node까지의 경로를 나타내는 이진트리 안에서 node의 vector를 업데이트한다.

이 방법의 속도는 이진트리가 어떻게 구성되고, 단어들이 leaf node들에 어떻게 할당되는가에 따라 결정된다.
Mikolov et al.은 자주 등장하는 단어들을 트리 안의 짧은 경로에 할당하는 binary Huffman tree를 사용했다.