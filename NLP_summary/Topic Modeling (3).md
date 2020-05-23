# Topic Modeling (3)

source: https://wikidocs.net/30707, https://wikidocs.net/24949, https://wikidocs.net/30708, https://wikidocs.net/40710

---

### 3. 잠재 디리클레 할당(LDA) 실습2

앞서 gensim을 통해서 LDA를 수행하고, 시각화를 진행해 보았다.
이번에는 사이킷런을 사용하여 LDA를 수행해 보자.
사이킷런을 사용하므로 전반적인 과정은 LSA 챕터와 유사하다.

---

### 3.1 실습을 통한 이해

1. 뉴스 기사 제목 데이터에 대한 이해
   약 15년 동안 발행되었던 뉴스 기사 제목을 모아놓은 영어 데이터를 아래 링크에서 받을 수 있다.

   - 링크: https://www.kaggle.com/therohk/million-headlines

   ```python
   import pandas as pd
   import urllib.request
   urllib.request.urlretrieve("https://raw.githubusercontent.com/franciscadias/data/master/abcnews-date-text.csv", filename="abcnews-date-text.csv")
   data = pd.read_csv('abcnews-date-text.csv', error_bad_lines=False)
   ```

   ```python
   print(len(data))
   ```

   ```python
   1082168
   ```

   해당 데이터는 약 100만개의 샘플을 갖고 있다.
   상위 5개의 샘플만 출력해보자.

   ```python
   print(data.head(5))
   ```

   ```python
      publish_date                                      headline_text
   0      20030219  aba decides against community broadcasting lic...
   1      20030219     act fire witnesses must be aware of defamation
   2      20030219     a g calls for infrastructure protection summit
   3      20030219           air nz staff in aust strike for pay rise
   4      20030219      air nz strike to affect australian travellers
   ```

   이 데이터는 publish_data와 headline_text라는 두 개의 열을 갖고 있다.
   각각 뉴스가 나온 날짜와 뉴스 기사 제목을 의미한다.
   필요한 데이터는 이 중에서 headline_text 열이다.
   즉, 뉴스 기사 제목이므로 아 부분만 별도로 저장한다.

   ```python
   text = data[['headline_text']]
   text.head(5)
   ```

   ```python
                                          headline_text
   0  aba decides against community broadcasting lic...
   1     act fire witnesses must be aware of defamation
   2     a g calls for infrastructure protection summit
   3           air nz staff in aust strike for pay rise
   4      air nz strike to affect australian travellers
   ```

   정상적으로 headline_text 열만 추출된 것을 확인할 수 있다.

2. 텍스트 전처리
   이번 챕터에서는 불용어 제거, 표제어 추출, 길이가 짧은 단어 제거라는 세 가지 전처리 기법을 사용한다.

   ```python
   import nltk
   text['headline_text'] = text.apply(lambda row: nltk.word_tokenize(row['headline_text']), axis=1)
   ```

   NLTK의 word_tokenize를 통해 단어 토큰화를 수행한다.

   ```python
   print(text.head(5))
   ```

   ```python
                                          headline_text
   0  [aba, decides, against, community, broadcastin...
   1  [act, fire, witnesses, must, be, aware, of, de...
   2  [a, g, calls, for, infrastructure, protection,...
   3  [air, nz, staff, in, aust, strike, for, pay, r...
   4  [air, nz, strike, to, affect, australian, trav...
   ```

   상위 5개의 샘플만 출력하여 단어 토큰화 결과를 확인한다.
   이제 불용어를 제거한다.

   ```python
   from nltk.corpus import stopwords
   stop = stopwords.words('english')
   text['headline_text'] = text['headline_text'].apply(lambda x: [word for word in x if word not in (stop)])
   ```

   여기서는 NLTK가 제공하는 영어 불용어를 통해서 text 데이터로부터 불용어를 제거해보자.

   ```python
   print(text.head(5))
   ```

   ```python
                                          headline_text
   0   [aba, decides, community, broadcasting, licence]
   1    [act, fire, witnesses, must, aware, defamation]
   2     [g, calls, infrastructure, protection, summit]
   3          [air, nz, staff, aust, strike, pay, rise]
   4  [air, nz, strike, affect, australian, travellers]
   ```

   상위 5개의 샘플에 대해서 불용어를 제거하기 전과 후의 데이터만 비교해도 확실히 몇 가지 단어들이 사라진 것이 보인다.
   against, be, of, a, in, to 등의 단어가 제거되었다.
   이제 표제어 추출을 수행한다.
   표제어 추출로 3인칭 단수 표현을 1인칭으로 바꾸고, 과거 현재형 동사를 현재형으로 바꾼다.

   ```python
   from nltk.stem import WordNetLemmatizer
   text['headline_text'] = text['headline_text'].apply(lambda x: [WordNetLemmatizer().lemmatize(word, pos='v') for word in x])
   print(text.head(5))
   ```

   ```python
                                          headline_text
   0       [aba, decide, community, broadcast, licence]
   1      [act, fire, witness, must, aware, defamation]
   2      [g, call, infrastructure, protection, summit]
   3          [air, nz, staff, aust, strike, pay, rise]
   4  [air, nz, strike, affect, australian, travellers]
   ```

   표제어 추출이 된 것을 확인할 수 있다.
   이제 길이가 3이하인 단어에 대해서 제거하는 작업을 수행한다.

   ```python
   tokenized_doc = text['headline_text'].apply(lambda x: [word for word in x if len(word) > 3])
   print(tokenized_doc[:5])
   ```

   ```python
   0       [decide, community, broadcast, licence]
   1      [fire, witness, must, aware, defamation]
   2    [call, infrastructure, protection, summit]
   3                   [staff, aust, strike, rise]
   4      [strike, affect, australian, travellers]
   ```

   길이가 3이하인 단어들에 대해서 제거가 된 것을 볼 수 있다.
   이제 TF-IDF 행렬을 만들어보자.

3. TF-IDF 행렬 만들기
   TfidfVectorizer는 기본적으로 토큰화가 되어있지 않은 텍스트 데이터를 입력으로 사용한다.
   이를 사용하기 위해 다시 토큰화 작업을 역으로 취소하는 역토큰화(Detokenization)작업을 수행해보자.

   ```python
   # 역토큰화 (토큰화 작업을 되돌림)
   detokenized_doc = []
   for i in range(len(text)):
       t = ' '.join(tokenized_doc[i])
       detokenized_doc.append(t)
   
   text['headline_text'] = detokenized_doc # 다시 text['headline_text']에 재저장
   ```

   역토큰화가 되었는지 text['headline_text']의 5개의 샘플을 출력해보자.

   ```python
   text['headline_text'][:5]
   ```

   ```python
   0       decide community broadcast licence
   1       fire witness must aware defamation
   2    call infrastructure protection summit
   3                   staff aust strike rise
   4      strike affect australian travellers
   Name: headline_text, dtype: object
   ```

   정상적으로 역토큰화가 수행되었음을 확인할 수 있다.
   이제 사이킷런의 TfidfVectorizer를 TF-IDF 행렬로 만들 것입니다.
   텍스트 데이터에 있는 모든 단어를 가지고 행렬을 만들 수도 있겠지만, 여기서는 간단히 1,000개의 단어로 제한한다.

   ```python
   from sklearn.feature_extraction.text import TfidfVectorizer
   vectorizer = TfidfVectorizer(stop_words='english', 
   max_features= 1000) # 상위 1,000개의 단어를 보존 
   X = vectorizer.fit_transform(text['headline_text'])
   X.shape # TF-IDF 행렬의 크기 확인
   ```

   ```python
   (1082168, 1000)
   ```

   1,082,168 × 1,000의 크기를 가진 가진 TF-IDF 행렬이 생겼다.
   이제 이에 LDA를 수행한다.

4. 토픽 모델링

   ```python
   from sklearn.decomposition import LatentDirichletAllocation
   lda_model=LatentDirichletAllocation(n_components=10,learning_method='online',random_state=777,max_iter=1)
   ```

   ```python
   lda_top=lda_model.fit_transform(X)
   ```

   ```python
   print(lda_model.components_)
   print(lda_model.components_.shape) 
   ```

   ```python
   [[1.00001533e-01 1.00001269e-01 1.00004179e-01 ... 1.00006124e-01
     1.00003111e-01 1.00003064e-01]
    [1.00001199e-01 1.13513398e+03 3.50170830e+03 ... 1.00009349e-01
     1.00001896e-01 1.00002937e-01]
    [1.00001811e-01 1.00001151e-01 1.00003566e-01 ... 1.00002693e-01
     1.00002061e-01 7.53381835e+02]
    ...
    [1.00001065e-01 1.00001689e-01 1.00003278e-01 ... 1.00006721e-01
     1.00004902e-01 1.00004759e-01]
    [1.00002401e-01 1.00000732e-01 1.00002989e-01 ... 1.00003517e-01
     1.00001428e-01 1.00005266e-01]
    [1.00003427e-01 1.00002313e-01 1.00007340e-01 ... 1.00003732e-01
     1.00001207e-01 1.00005153e-01]]
   (10, 1000)
   ```

   ```python
   terms = vectorizer.get_feature_names() # 단어 집합. 1,000개의 단어가 저장됨.
   
   def get_topics(components, feature_names, n=5):
       for idx, topic in enumerate(components):
           print("Topic %d:" % (idx+1), [(feature_names[i], topic[i].round(2)) for i in topic.argsort()[:-n - 1:-1]])
   get_topics(lda_model.components_,terms)
   ```

   ```python
   Topic 1: [('government', 8725.19), ('sydney', 8393.29), ('queensland', 7720.12), ('change', 5874.27), ('home', 5674.38)]
   Topic 2: [('australia', 13691.08), ('australian', 11088.95), ('melbourne', 7528.43), ('world', 6707.7), ('south', 6677.03)]
   Topic 3: [('death', 5935.06), ('interview', 5924.98), ('kill', 5851.6), ('jail', 4632.85), ('life', 4275.27)]
   Topic 4: [('house', 6113.49), ('2016', 5488.19), ('state', 4923.41), ('brisbane', 4857.21), ('tasmania', 4610.97)]
   Topic 5: [('court', 7542.74), ('attack', 6959.64), ('open', 5663.0), ('face', 5193.63), ('warn', 5115.01)]
   Topic 6: [('market', 5545.86), ('rural', 5502.89), ('plan', 4828.71), ('indigenous', 4223.4), ('power', 3968.26)]
   Topic 7: [('charge', 8428.8), ('election', 7561.63), ('adelaide', 6758.36), ('make', 5658.99), ('test', 5062.69)]
   Topic 8: [('police', 12092.44), ('crash', 5281.14), ('drug', 4290.87), ('beat', 3257.58), ('rise', 2934.92)]
   Topic 9: [('fund', 4693.03), ('labor', 4047.69), ('national', 4038.68), ('council', 4006.62), ('claim', 3604.75)]
   Topic 10: [('trump', 11966.41), ('perth', 6456.53), ('report', 5611.33), ('school', 5465.06), ('woman', 5456.76)]
   ```

   

