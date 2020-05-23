# **What's the NLP? (1) :thinking:**

이 글은 스탠포드 대학에서 제공하는 딥러닝을 이용한 NLP -CS224n- 강의노트(Lecture Note) 1장의 앞 부분을 번역한 내용입니다.
또한 이 글은 "솔라리스의 인공지능 연구실"의 글을 참조하여 작성되었습니다.
Source: http://solarisailab.com/archives/818

Reference
[1] http://web.stanford.edu/class/cs224n/syllabus.html

[2] http://web.stanford.edu/class/cs224n/lecture_notes/cs224n-2017-notes1.pdf

[3] https://arxiv.org/pdf/1301.3781.pdf

---

**1.1 What is so special about NLP?**
인간의 언어는 어떤 종류의 물리적인 표시를 위해서가 아닌, 의미를 전달하기 위해 고안된 시스템이다.
따라서, vision이나 다른 machine learning의 task들과는 매우 다르다.

대부분의 단어들이 언어학적인 실체(linguist entity)와 관련 없는 단순한 상징(symbol)이다.
단어는 기의시니피에(아이디어나 어떤 의미)에 기표시니피앙을 맵핑(map)한 것이다.

예를 들어, "rocket"이란 단어는 로켓의 컨셉을 지칭하고, 이를 확장하여 로켓의 실체를 가르킨다.
하지만, 몇가지의 예외들도 있다.
예를 들어, 우리가 감탄을 나타내는 단어와 문자로써 "Whooompaa"와 같은 표현을 사용할 때가 있다.
게다가, 언어의 상징들은 여러가지 양식(modalities)으로 부호화(encoded)가 될 수 있다.
이런 양식들(음성, 몸짓, 쓰기 등)은 연속적인 신호들(continuous signals)로 뇌에 전달된다.
따라서, 연속적인 형태로 부호화 된다.

**1.2 Examples of tasks**
음성처리(Speech Processing)부터 의미해석(Sematic Interpretation), 담화분석(Discourse Processing)등의 다양한 난이도(level)의 NLP의 문제들이 있다.
NLP의 목적은 컴퓨터가 어떤 작업을 수행하기 위해서 자연어를 "이해"하는 알고리즘을 디자인하는 것이다.
예제들은 난이도에 따라 분류될 수 있다.

- Easy
  1. 철자 검사(Spell Checking)
  2. 키워드 검색(Keyword Search)
  3. 동의어 찾기(Finding Synonyms)
- Medium
  - 웹사이트, 문서 등에서 정보 추출(Parsing information from websites, documents, etc...)
- Hard
  1. 기계 번역(Machine Translation)
  2. 의미 분석(Sematic Analysis)
  3. 동일 지시어(Coreference)
  4. 질의 응답(Question Answering)

**1.3 How to represent words?**
틀림없이 거의 모든 NLP의 task들은 통틀어서 가장 중요한 공통 분모는 입력으로 들어오는 단어들을 다른 모델로 어떻게 나타낼 것 인가에 관한 문제이다.
대부분의 초기 NLP 연구들은 단어들을 원자 기호들(atom symbols)로 나타내었다.
NLP 문제들을 잘 풀기 위해서는, 첫번째로 단어들간의 유사성과 차이에 대한 개념(notion)이 있어야만 한다.
word vector들을 통해서, 우리는 백터 그 자체를 이용해서 이런 특징을 손쉽게 가져갈 수 있다.

---

**2. Word Vectors**
영어에는 약 13,000,000rodml 토큰들(tokens)이 존재할 것으로 추정된다.
하지만, 그들이 전혀 서로 연관관계가 없을까?
고양이과 동물(feline)과 고양이(cat), 호텔(hotel)과 모텔(motel)?
나는 그렇게 생각하지 않는다.
그러므로, 우리는 일종의 "word" space에 한 점을 나타내는 어떤 벡터로 각각의 단어 토큰들을 부호화할 수 있다.
이는 여러가지 이유로 매우 중요하지만, 아마도 가장 직관적인 이유는, N차원의 space (N<<13,000,000)가 우리 언어의 모든 의미를 부호화하기에 충분하다는 점이다.
각각의 차원은 우리가 말을 통해 전달하는 의미를 부호화한다.
예를 들어, 의미 차원들(semantic dimensions)은 시제(tense), 수치, 성 등을 지칭할 수 있다.

따라서, 우리의 첫 word vectors이자, 가장 간단한 방법인 one-hot vector를 살펴보자.
이는 모든 단어를 정렬된 영어 사전의 index에 해당되면 **1**을 그렇지 않은 모든 곳에 **0**을 할당한 ![R^{|V|\times{M}}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-c37fb256e8b68f1b1affcb0cd2dacca0_l3.png) 벡터로 나타낸다.
이 표현을 따르면, ![|V|](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-c40bbcc0a59a786cf47412de36ed7c9f_l3.png)는 우리의 어휘 사전의 크기이다.
이런 encoding을 통한 word vectors는 아래와 같은 형태를 뛸 것이다.
![\begin{equation*} W^{aardvark}= \begin{bmatrix} 1 \\ 0 \\ 0 \\ \vdots \\ 0 \end{bmatrix}, W^{a}= \begin{bmatrix} 0 \\ 1 \\ 0 \\ \vdots \\ 0 \end{bmatrix}, W^{at}= \begin{bmatrix} 0 \\ 0 \\ 1 \\ \vdots \\ 0 \end{bmatrix}, W^{zebra}= \begin{bmatrix} 0 \\ 0 \\ 0 \\ \vdots \\ 1 \end{bmatrix} \end{equation*}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-07f221f8fff7c4cc0425ca499f7e769c_l3.png)

우리는 각각의 단어를 완전히 독립적인 엔티티(entity)로 나타낼 것이다.
전에 언급했듯이, 이런 단어 표현은 유사성에 대한 개념을 직접적으로 나타내지 않는다.
예를 들어 다음과 같이, 역자주 hotel-motel은 유사성이 있는 단어쌍이지만 hotel-cat 단어쌍과 곱했을 때 같은 결과값을 갖는다.

![\begin{equation*} (w^{hotel})^T{w^{motel}}=(w^{hotel})^T{w^{cat}}=0 \end{equation*}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-d62918f26686b2df1e39b6771eaea415_l3.png)

그러므로, 우리는 space의 크기를 ![R|V|](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-48425a1c6bb5ca9c65e262165587a950_l3.png)에서 좀 더 작은 크기로 축소할 수 있을 것이다.
또한, 단어들간의 연관성을 나타내는 subspace를 encode 할 수 있을 것이다.

---

**3. SVD Based Methods**
word embeddings(word vectors)을 찾기 위해서, 첫번째로, 대량의 데이터셋을 돌면서, 단어가 동시에 등장(co-occurrence)하는 횟수를 어떤 matrix ![X](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-28119da568472672b4bea9c2bd76c89a_l3.png)형태로 수집한다.
그리고 나서 ![USV^T](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-4956599568759976faff82d6710ae80e_l3.png)decomposition을 얻기 위해 ![X](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-28119da568472672b4bea9c2bd76c89a_l3.png)에 대해 Singular Value Decomposition을 수행한다.
그리고 나서, ![U](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-02cfda02c2fd09347e3f499c66782b47_l3.png)의 행들(rows)을 사전에 존재하는 모든 단어들의 word embedding으로 사용한다.
이제 ![X](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-28119da568472672b4bea9c2bd76c89a_l3.png)를 모델링하기 위한 몇가지 방법들을 논의해보자.

**3.1 Word-Document matrix**
우리의 첫번째 시도로, 같은 문서에서 나타나는 연관된 단어들에 대한 추측을 진행할 것이다.
예를 들어, "bank", "bonds", "stocks", "moneys"등은 같이 등장할 확률이 높을 것이다.
하지만 "banks", "octopus", "banana", "hockey"등은 같이 등장할 확률이 낮을 것이다.
우리는 이런 사실을 word-matrix ![X](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-28119da568472672b4bea9c2bd76c89a_l3.png)를 만드는데 다음과 같은 방법을 이용해서 사용할 것이다.
많은 양의 문서들을 흝어가면서 문서 ![j](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-f565589cb98b3eaf1dfd6ad0a48aafcd_l3.png)에 단어 ![i](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-cc64060f364aab1f0e662c8a8a0816c4_l3.png)가 등장할 때마다, ![X_{ij}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-2a688f3be0995b73c2d6c31273c503a2_l3.png)엔트리(entry)애 1을 더한다.
이는 명백히 매우 큰 행렬(![R^{|V|\timesM}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-8ecd48fc4df61b840d38c59486397ec1_l3.png))을 만들게 되고, 행렬의 크기는 문서의 수 ![(M)](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-2c0c5f00eb1e8240d8954cb1738bd3fa_l3.png)에 비례하여 커지게 된다.
그러므로 우리는 더 나은 방법을 생각해야 한다.

**3.2 Window based Co-occurrence Matrix**
같은 논리(logic)를 Window based Co-occurrence Matrix에 적용할 수 있다.
하지만, 행렬 ![X](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-28119da568472672b4bea9c2bd76c89a_l3.png)가 단어의 동시 발생을 저장하면 유사도 행렬(affinity matrix)이 된다.
이 방법은 관심있는 단어의 주변에 특정한 사이즈의 윈도우를 지정하고, 각 단어가 윈도우 내에서 몇번이나 등장하는지를 센다.
corpus에 있는 모든 단어들에 대해 이런 count를 계산한다.
아래의 예제를 참조하자.
우리의 corpus가 오직 3개의 문장만을 가지고 있고, window size를 1로 지정했을 경우를 가정하자.

1. I enjoy flying
2. I like NLP
3. I like deep learning

그렇다면, 결과로 출력되는 matrix는 아래와 같은 형태일 것이다.

![\begin{equation*} W= \bordermatrix{ & I     & like     & enjoy & deep & learning & NLP & flying & .   \cr I         & 0   & 2  & 1 & 0 & 0 & 0 & 0 &0    \cr like    & 2   & 0  & 0 & 1 & 0 & 1 & 0 &0     \cr enjoy & 1  & 0  & 0 & 0 & 0 & 0 & 1 &0  \cr deep  & 0  & 1  & 0 & 0 & 1 & 0 & 0 &0  \cr learning & 0  & 0  & 0 & 1 & 0 & 0 & 0 &1 \cr NLP & 0  & 1  & 0 & 0 & 0 & 0 & 0 &1 \cr flying & 0  & 0  & 1 & 0 & 0 & 0 & 0 &1 \cr . & 0  & 0  & 0 & 0 & 1 & 1 & 1 &0 \cr } \end{equation*}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-dd0e11dfa9a41cc4a63126f70f9b3f78_l3.png)

**3.3 Applying SVD to the co-occurrence matrix**
이제 행렬 ![X](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-28119da568472672b4bea9c2bd76c89a_l3.png)에 대해 SVD를 수행하자.
singular value들을 관찰하고(출력되는 ![S](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-acc351b628524005c097e4b643da021d_l3.png)행렬의 대각선 값들(entries)), 포착하고자 하는 비율의 분산(percentage variance)에 기반해 설정한 인덱스 ![k](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-73b9da2a6ea70ff45777a87fdc467394_l3.png)를 이용해서 그들을 자른다.

![\begin{equation*} \frac{\sum_{i=1}^{k}\sigma_i}{\sum_{i=1}^{|V|}\sigma_i} \end{equation*}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-a3d56d724841041a805496b5ce173774_l3.png)

그 다음에 word embedding matrix를 얻기 위해서 submatrix ![U_{1:|V|,1:k}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-fdc1cc67ddff265e816f4667cd8aa7cc_l3.png)를 구한다.
이 방법은 어휘 사전안에 모든 단어들을 ![k](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-73b9da2a6ea70ff45777a87fdc467394_l3.png)-차원 표현으로 나타낼 것이다.

- Applying SVD to ![X](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-28119da568472672b4bea9c2bd76c89a_l3.png):

  ![\begin{equation*} \bordermatrix{ &   & |V|  & \cr & & & \cr |V|   &  & X &  \cr & & & \cr }= \bordermatrix{ &   & |V|  & \cr &| &| & \cr |V|   & u_1 & u_2 & \cdots \cr & | & | &   \cr } \bordermatrix{ &   & |V|  & \cr &\sigma_1 &0 & \cdots \cr |V|   & 0 &\sigma_2 & \cdots \cr & \vdots & \vdots & \ddots  \cr } \bordermatrix{ &   & |V|  & \cr &\ -& v_1 & - \cr |V|   & - & v_2 & - \cr &  & \vdots &   \cr } \end{equation*}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-c9d617b3a321d8b31b720b2cf4e4e99c_l3.png)

- 첫 ![k](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-73b9da2a6ea70ff45777a87fdc467394_l3.png)개의 singular 벡터만을 선택해서 차원을 축소하기(Reducing dimensionality by selecting first ![k](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-73b9da2a6ea70ff45777a87fdc467394_l3.png) singular vectors):

  ![\begin{equation*} \bordermatrix{ &   & |V|  & \cr & & & \cr |V|   &  & \hat{X} &  \cr & & & \cr }= \bordermatrix{ &   & k  & \cr &| &| & \cr |V|   & u_1 & u_2 & \cdots \cr & | & | &   \cr } \bordermatrix{ &   & k  & \cr &\sigma_1 &0 & \cdots \cr k   & 0 &\sigma_2 & \cdots \cr & \vdots & \vdots & \ddots  \cr } \bordermatrix{ &   & |V|  & \cr &\ -& v_1 & - \cr k   & - & v_2 & - \cr &  & \vdots &   \cr } \end{equation*}](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-3e8672964b2df776c61d8db645ccc933_l3.png)

  두 방법 모두 의미적(semantic), 구문적(syntactic) 정보를 부호화하기에 충분한 word vectors를 만들게 해준다.
  하지만 여전히 많은 문제들이 남아있다.

  - 행렬의 차원이 빈번하게 바뀐다. (새로운 단어가 매우 빈번하게 추가되고, corpus의 크기가 바뀐다.)
  - 행렬이 매우 sparse하다. (대부분의 단어들이 동시발생(co-occur)하지 않기 때문)
  - 일반적으로 행렬이 매우 고차원이다. (![\approx 10^6\times10^6](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-d5989815fb93037de73fb6178e5cc8e6_l3.png))
  - 학습하는데 quadratic cost가 필요하다. (i.e. SVD를 수행하기 위해서)
  - 단어 빈도에 대한 극도의 불균형을 보정하기 위해서 ![X](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-28119da568472672b4bea9c2bd76c89a_l3.png)에 대한 조작(hack)이 필요하다.

위에 나열된 문제점들을 해결하기 위한 몇가지 방법들이 있다.

- 기능어(function words) "the", "he", "has"등을 무시한다.
- ramp window를 적용한다. (i.e. 문서 안에서 단어들간의 거리(distance)에 기반해서 co-occurrence 가중치(weight)를 준다.
- Pearson correlation을 사용하고 단순한 카운트(raw count) 대신에 negative counts를 사용한다.