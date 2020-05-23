# Text Processing (1)

Source: https://wikidocs.net/21694, https://wikidocs.net/21698, https://wikidocs.net/21693, https://wikidocs.net/21707, https://wikidocs.net/22530, https://wikidocs.net/21703, https://wikidocs.net/31766, https://wikidocs.net/22647, https://wikidocs.net/22592, https://wikidocs.net/33274
NLP에 있어서 Text Processing은 매우 중요한 작업이다.

---

### 1. 토큰화(Tokenization)

NLP에서 크롤링 등으로 통해서 얻어낸 커퍼스 데이터가 필요에 맞게 전처리되지 않은 상태이라면, 해당 데이터를 사용하고자 하는 용도에 맞게 토큰화(tokenization), 정제(cleaning), 정규화(normalization)하는 일을 하게된다.

주어진 코퍼스(corpus)에서 토큰이라고 불리는 단위로 나누는 작업을 토콘화라고 한다.
토큰의 단위는 상황에 따라 다르지만, 보통 의미있는 단위로 토큰을 정의한다.

---

### 1.1 단어 토큰화(Word Tokeniaztion)

토큰의 기준을 단어로 하는 경우, 이를 단어 토큰화하고 한다.
다만, 여기서 단어는 단위 외에도 단어구, 의미를 갖는 문자열로도 간주되기도 한다.

예를 들어보자면, 아래의 입력으로부터 구두점(punctuation)과 같은 문자는 제외시키는 간단한 단어 토큰화 작업한다고 가정해보자.
*구두점: 온점(.), 컴마(,), 물음표(?), 세미클론(;), 느낌표(!) 등과 같은 기호

입력: Time is an illusion. Lunchtime double so!

입력의 구두점을 제외시키고 토큰화 작업을 한 결과는 다음과 같다.

출력: "Time", "is", "an", "illusion", "Lunchtime", "double", "so"

이 예제에서 토큰화 작업은 굉장히 간단하다.
구두점을 지운 뒤에 whitespace를 기준으로 잘라낸 결과다.
하지만 이 예제는 토큰화의 가장 기초적인 예제에 불과하다.

보통 토큰화 작업은 단순히 구두점이나 특수문자를 전부 제거하는 정제 작업을 수행하는 것만으로 해결되지 않는다.
구두점이나 특수문자를 전부 제거하면 토큰이 의미를 잃어버리는 경우가 발생하기도 한다.
심지어 whitespace 단위로 자르면 사실상 단어 토큰이 구분되는 영어와 달리, 한국어는 whitespace만으로는 구분하기 어렵다.

---

### 1.2 토큰화 중 생기는 선택의 순간

토큰화를 진행하다보면, 예상하지 못한 경우가 발생하여 토큰화의 기준을 생각해야하는 경우가 발생한다.
이러한 선택은 해당 데이터를 가지고 어떤 용도로 사용할 것인지에 따라, 그 용도에 영향이 없는 기준으로 정하면 된다.
영어권 언어에서 아포스트로피(')가 들어가 있는 단어는 어떻게 토큰으로 분류할지에 대한 문제를 예시로 들어보자.

Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop.

아포스트로피가 들어간 상황에서 Don't와 Jone's는 어떻게 토큰화할 수 있는가?

- Don't
- Don t
- Dont
- Do n't

- Jone's
- Jone s
- Jone
- Jones

원하는 결과가 나오도록 토큰화 도구를 직접 설계할 수도 있지만, 기존에 공개된 도구들을 사용했을 때의 결과가 사용자의 목적과 일치한다면 해당 도구를 사용할 수도 있다.
NLTK는 영어 코퍼스를 토큰화하기 위한 도구들을 제공한다.
그 중 `word_tokenize`와 `WordPunctTokenizer`를 사용해서 NTLK에서는 아포스트로피를 어떻게 처리하는지 확인해보자.

```python
from nltk.tokenize import word_tokenize`
print(word_tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
```

```python
['Do', "n't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', ',', 'Mr.', 'Jone', "'s", 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop', '.']
```

`word_tokenize`는 Don't를 n't로 분리했으며, Jone's는 Jone 's로 분리했다.

`WordPunctTokenizer`는 다음과 같이 처리한다.

```python
from nltk.tokenize import WordPunctTokenizer`
print(WordPunctTokenizer().tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
```

```python
['Don', "'", 't', 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', ',', 'Mr', '.', 'Jone', "'", 's', 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop', '.']
```

`WordPunctTokenizer`는 구두점을 별도로 분류하는 특징을 갖고 있기 때문에, 앞서 확인 했던 `word_tokenize`와는 달리 Don't를 Don과 '와 t로 분리하였으며, 이와 마찬가지로 Jone's를 Jone과 '와 s로 분리했다.

또한 케라스에도 토큰화 도구로서 `text_to_word_sequence`를 지원하는데, 이는 기본적으로 모든 알파벳을 소문자로 바꾸면서 온점이나 컴마, 느낌표 등의 구두점을 제거한다.
하지만 don't나 Jone's와 같은 경우 아포스트로피는 보존한다.

---

### 1.3 토큰화에서 고려해야 할 사항

토큰화 작업을 단순하게 코퍼스에서 구두점을 제외하고 공백 기준으로 잘라내는 작업이라고 간주할 수 없다.
이러한 일은 보다 섬세한 알고리즘이 필요하다.

1. 구두점이나 특수문자를 단순 제외해서는 안된다.
   갖고있는 코퍼스에서 단어들을 걸러낼 때, 구두점이나 특수 문자를 단순히 제외하는 것은 옳지 않다.
   코퍼스에 대한 정제 작업을 진행하다보면, 구두점조차도 하나의 토큰으로 분류하기도 한다.
   가장 기본적인 예를 들어보자면, 온점(.)과 같은 경우는 문장의 경계를 알 수 있는데 도움이 되므로 단어를 뽑아낼 때, 온점(.)을 제외하지 않을 수 있다.

   또 다른 예를 들어보면, 단어 자체에서 구두점을 갖고 있는 경우도 있다.
   m.p.h나 Ph.D나 AT&T 같은 경우가 있다.
   또 특수문자의 달러($)나 슬래시(/)로 예를 들어보면, $45.55와 같은 가격을 의미하기도 하고, 01/02/06은 날짜를 의미하기도 한다.
   보통 이런 경우 45.55를 하나로 취급할려고 한다.

   숫자 사이에 컴마(,)가 들어가는 경우도 있다.
   가령 보통 수치를 표현할 때는 123,456,789와 같이 세 자리 단위로 컴마가 들어간다.

2. 줄임말과 단어 내에 띄어쓰기가 있는 경우
   토큰화 작업에서 종종 영어권 언어의 아포스트로피(')는 압축된 단어를 다시 펼치는 역할을 하기도 한다.
   예를 들어, what're는 what are의 줄임말이며 we're는 we are의 줄임말이다.
   위의 예에서 re를 접어(clitic)이라고 한다.
   단어가 줄임말로 쓰일 때 생기는 형태를 지칭한다.
   가령 I am를 줄인 I'm이 있을 때, m을 접어라고 한다.

   New York라는 단어나 rock 'n' roll이라는 단어는 하나의 단어이지만, 중간에 띄어쓰가 존재한다.
   사용 용도에 따라, 하나의 단어 사이에 띄어쓰기가 있는 경우에도 하나의 토큰으로 봐야하는 경우도 있을 수 있으므로, 토큰화 작업은 이러한 단어를 하나로 인식할 수 있는 능력을 가져야 한다.

3. 표준 토큰화 예제
   이해를 위해, 토큰화 방법 중 하나인 Penn Treebank Tokenization의 규칙에 대해서 보자.

   - 규칙1. 하이푼으로 구성된 단어는 하나로 유지한다.
   - 규칙2. doesn't와 같이 아포스트로피로 접어가 함께하는 단어는 분리한다.

   해당 표준에 아래의 문장을 input 해보자.

   "Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own."

   ```python
   from nltk.tokenize import TreebankWordTokenizer
   tokenizer=TreebankWordTokenizer()
   text="Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own."

   print(tokenizer.tokenize(text))
   ```
```
   
   ```python
   ['Starting', 'a', 'home-based', 'restaurant', 'may', 'be', 'an', 'ideal.', 'it', 'does', "n't", 'have', 'a', 'food', 'chain', 'or', 'restaurant', 'of', 'their', 'own', '.']
```

   결과를 보면, 각각 규칙 1과 규칙 2에 따라 home-based는 하나의 토큰으로 취급하고 있으며, dosen't의 경우 does와 n't로 분리 됬다.

---

### 1.4 문장 토큰화(Sentence Tokenization)

토큰의 단위가 문장일 때, 토큰화는 다음과 같이 진행된다.
이 작업은 갖고있는 코퍼스 내에서 문장 단위로 구분하는 작업으로 때로는 문장 분류(sentence segmentation)라고 부른다.

보통 갖고있는 코퍼스가 정제되지 않은 상태라면, 코퍼스는 문장 단위로 구분되어있지 않을 가능성이 높다.
이를 사용하고자 하는 용도에 맞게 하기 위해서는 문장 토큰화가 필요할 수도 있다.

직관적으로  생각해봤을 때, 물음표(?)나 온점(.)이나 느낌표(!) 기준으로 문장을 잘라내면 되지 않을까라고 생각할 수 있지만, 꼭 그렇지만은 않는다.
!나 ?는 문장의 구분을 위한 꽤 명확한 구분자(boundary) 역할을 수행하지만, 온점(.) 꼭 그렇지 않는다.
다시 말해, 온점(.)은 문장의 끝이 아니더라도 문장의 중간에 등장할 수 있다.

- Ex1) IP 192.168.56.31 서버에 들어가서 로그 파일 저장해서 ukairia777@gmail.com로 결과 좀 보내줘. 그러고나서 점심 먹으러 가자.
- Ex2) Since I'm actively looking for Ph.D. students, I get the same question a dozen times every year.

예시로 위의 예제들에 온점을 기준으로 문장 토큰화를 적용하면 어떻게 될까.
첫번째 예시에서 '보내줘.'에서 그리고 두번째 예제에서는 'year.'에서 처음으로 문장이 끝난 것으로 인식하는 것이 제대로 문장의 끝을 예측했다고 볼 수 있을 것이다.
하지만 단순히 온점(.)으로 문장을 구분짓는다고 가정하면, 문장의 끝이 나오기 전에 이미 온점이 여러번 등장하여 예상한 결과가 나오지 않게 된다.

그렇기에 사용하는 코퍼스가 어떤 국적의 언어인지 또는 코퍼스 내에서 특수문자들이 어떻게 사용되고 있는지에 따라서 직접 규칙들을 정의해볼 수 있다.
물론 100%의 정확도를 얻는 것은 쉬운 일이 아니다.
갖고 있는 코퍼스 데이터에 오타나, 문장의 구성이 엉망이라면 정해놓은 규칙이 소용이 없을 수 있기 때문이다.

NLTK에서는 영어 문장의 토큰화를 수행하는 sent_tokenize를 지원하고 있다.

```python
from nltk.tokenize import sent_tokenize
text="His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to mae sure no one was near."

print(sent_tokenize(text))
```

```python
['His barber kept his word.', 'But keeping such a huge secret to himself was driving him crazy.', 'Finally, the barber went up a mountain and almost to the edge of a cliff.', 'He dug a hole in the midst of some reeds.', 'He looked about, to mae sure no one was near.']
```

위 코드는 text에 저장된 여러개의 문장들로 부터 문장을 구분하는 코드이다.
출력 결과를 보면 성공적으로 모든 문장을 구분해내었음을 볼 수 있다.
그렇다면, 이번에는 언급했던 문장 중간에 온점이 여러번 등장하는 경우에 대해서도 실습을 해보자.

```python
from nltk.tokenize import sent_tokenize
text="I am actively looking for Ph.D. students. and you are a Ph.D student."

print(sent_tokenize(text))
```

```python
['I am actively looking for Ph.D. students.', 'and you are a Ph.D student.']
```

NLTK는 단순히 온점을 구분자로 하여 문장을 구분하지 않았기 땨문에, Ph.D.를 문장 내의 단어로 인식하여 성공적으로 인식하는 것을 볼 수 있다.

물론, 한국어에 대한 문장 토큰화 도구가 존재한다.

---

### 1.5 이진 분류기(Binary Classifier)

문장 토큰화에서의 예외 사항을 발생시키는 온점(.)의 처리를 위해서 입력에 따라 두 개의 클래스로 분류하는 이진 분류기를 사용하기도 한다.

물론, 여기서 말하는 두 개의 클래스는

1. 온점(.)이 단어의 일부분일 경우. 즉, 온점이 약어(abbreivation)로 쓰이는 경우
2. 온점(.)이 정말로 문장의 구분자일 경우를 의미한다.

이진 분류기는 임의로 정한 여러가지 규칙을 코딩한 함수일 수도 있으며, ML을 통해 이진 분류기를 구현하기도 한다.

온점(.)이 어떤 클래스에 속하는지 결정하기 위해서 어떤 온점이 주로 약어(abbreviation)으로 쓰이는 지 알아야한다.
그렇기에, 이진 분류기 구현에서 약어 사전(abbreviation dictionary)를 유용하게 쓰인다.
*참고: https://tech.grammarly.com/blog/posts/How-to-Split-Sentences.html

---

### 1.6 한국어에서의 토큰화의 어려움

영어는 New York와 같은 합성어나 he's와 같이 줄임말에 대한 예외처리만 한다면, whitespace를 기준으로 하는 띄어쓰기 토큰화를 수행해도 단어 토큰화가 잘 수행된다.
거의 대부분의 경우에서 단어 단위로 whitespace가 이루어지기 때문에 whitespace 토큰화와 단어 토큰화가 거의 같기 때문이다.

하지만 한국어는 영어와는 달리 whitespace만으로는 토큰화 하기에는 부족하다.
한국어의 경우에는 whitespace 단위가 되는 단위를 '어절'이라고 하는데, 어절 토큰화는 한국어 NLP에서 지양되고 있다.
어절 토큰화와 단어 토큰화가 같지 않기 때문이다.
그 근본적인 이유는 한국어가 영어와는 다른 형태를 가지는 언어인 교착어라는 점에서 기인한다.
*교착어: 조사, 어미등을 붙여서 말을 만드는 언어를 말한다.

1. 한국어는 교착어이다.
   좀 더 자세히 설명하기 전에, 간단한 예를 들어보자.
   영어와는 달리 한국어에는 조사라는 것이 존재한다.
   예문으로, 그(he/him)라는 주어나 목적어가 들어간 문장이 있다고 가정하자.
   이 경우에는, 그라는 단어 하나에도 '그가', '그에게', '그를', '그와', '그는'과 같이 다양한 조사가 '그'라는 글자 뒤에 whitespace 없이 바로 붙게된다.
   NLP를 하다보면 같은 단어임에도 서로 다른 조사가 붙어서 다른 단어로 인식이 되면 NLP가 힘들고 번거로워지는 경우가 많다.
   대부분의 한국어 NLP에서 조사는 분리해줄 필요가 있다.

   띄어쓰기가 단위가 영어처럼 독립적인 단어라면 띄어쓰기 단위로 토큰화를 하면 되겠지만, 한국어는 어절이 독립적인 단어로 구성되는 것이 아니라 조사 등의 무언가가 붙어있는 겨우가 많아서 이를 전부 분리해줘야 한다는 의미이다.

   좀 더 자세하게 설명하자면, 한국어 토큰화에서는 형태소(morpheme)란 개념을 반드시 이해해야 한다.
   *형태소: 뜻을 가진 가장 작은 말의 단위
   이 형태소에는 두 가지 형태소가 있는데 자립 형태소와 의존 형태소이다.

   - 자립 형태소: 접사, 어미, 조사와 상관없이 자립하여 사용할 수 있는 형태소.
     그 자체로 단어가 된다. 체언(명사, 대명사, 수사), 수식언(관형사, 부사), 감탄사 등이 있다.
   - 의존 형태소: 다른 형태소와 결합하여 사용되는 형태소, 접사, 어미, 조사, 어간를 말한다.

   예를 들어 다음과 같은 예문이 있다고 가정하자.

   - 에디가 딥러닝책을 읽었다.

   이를 형태소 단위로 분해하면 다음과 같다.

   - 자립 형태소: 에디, 딥러닝책
   - 의존 형태소: -가, -을, 읽-, -었, -다

   이를 통해 유추할 수 있는 것은 한국어에서 영어에서의 단어 토큰화와 유사한 형태를 얻으려면 어절 토큰화가 아니라 형태소 토큰화를 수행해야한다는 것이다.

2. 한국어는 띄어쓰기가 영어보다 잘 지켜지지 않는다.
   사용하는 한국어 코퍼스가 뉴스 기사와 같이 띄어쓰기를 철저하게 지키려고 노력하는 글이라면 좋겠지만, 많은 경우 중 띄어쓰기를 틀렸거나, 지켜지지 않는 코퍼스가 많다.

   한국어는 영어권 언어와 비교하여 띄어쓰기가 어렵고, 또 잘 지켜지지 않는 경향이 있다.
   그 이유에는 여러 견해가 있는데, 가장 기본적인 견해는 한국어의 경우 띄어쓰기가 지켜지지 않아도 글을 쉽게 이해할 수 있는 언어라는 점이다.
   사실, 띄어쓰기가 없던 한국어에 띄어씌가 보편된 것도 근대(1933년)의 일이다.

   - Ex1) 제가이렇게띄어쓰기를전혀하지않고글을썼다고하더라도글을이해할수있습니다.
   - Ex2)  Tobeornottobethatisthequestion

   반면, 영어의 경우에는 띄어쓰기를 하지 않으면 알아보기가 어려운 문장들이 생긴다.
   이는 한국어(모아쓰기 방식)와 영어(풀어쓰기 방식)라는 언어적 특성의 차이에 기인한다.
   결론적으로 한국어는 수많은 코퍼스애서 띄어쓰기가 무시되는 경우가 많아 NLP가 어려워졌다는 것이다.

---

### 1.7 품사 태깅(Part-of-speech tagging)

단어는 표기는 같지만, 품사에 따라서 단어의 의미가 달라지기도 한다.
예를 들어서 영어 단어 'fly'는 동사로는 '날다'라는 의미를 갖지만, 명사로는 '파리'라는 의미를 갖고있다.
한국어도 마찬가지다.
'못'이라는 단어는 명사로서는 망치를 사용해서 목재 따위를 고정하는 물건을 의미한다.
하지만 부사로서의 '못'은 '먹는다', '달린다'와 같은 동작 동사를 고정하는 물건을 의미로 쓰인다.
즉, 결국 단어의 의미를 제대로 파악하기 위해서는 해당 단어가 어떤 품사로 쓰였는지 보는 것이 주요 지표가 될 수도 있다.
그에 따라 단어 토큰화 과정에서 각 단어가 어떤 품사로 쓰였는지를 구분해놓기도 하는데, 이 작업을 품사 태깅(part-of-speech tagging)이라고 한다.

---

### 1.8 NLTK와 KoNLP를 이용한 영어, 한국어 토큰화 실습

NLTK애서는 영어 코퍼스에 품사 태깅 기능을 지원한다.
품사를 어떻게 명명하고, 태깅하는지의 기준은 여러가지가 있는데, NLTK에서는 Penn Treebank POS Tags라는 기준을 사용합니다.
실제로 NLTK를 사용해서 영어 코퍼스에 품사 태깅을 해보도록 하자.

```python
from nltk.tokenize import word_tokenize
text="I am actively looking for Ph.D. students. and you are a Ph.D. student.

print(word_tokenize(text)
```

```python
['I', 'am', 'actively', 'looking', 'for', 'Ph.D.', 'students', '.', 'and', 'you', 'are', 'a', 'Ph.D.', 'student', '.']
```

```python
from nltk.tag import pos_tag
x=word_tokenize(text)
pos_tag(x)
```

```python
[('I', 'PRP'), ('am', 'VBP'), ('actively', 'RB'), ('looking', 'VBG'), ('for', 'IN'), ('Ph.D.', 'NNP'), ('students', 'NNS'), ('.', '.'), ('and', 'CC'), ('you', 'PRP'), ('are', 'VBP'), ('a', 'DT'), ('Ph.D.', 'NNP'), ('student', 'NN'), ('.', '.')]
```

영어 문장에 대해서 토큰화를 수행하고, 이어서 품사 태깅을 수행했다.
Penn Treebank POG Tags에서 PRP는 인칭 대명사, VBP는 동사, RB는 부사, VBG는 현재부사, IN은 전치사, NNP는 고유 명사, NNS는 복수형 명사, CC는 접속사, DT는 관사를 의미한다.

한국어 NLP를 위해서는 KoNLPy라는 파이썬 패키지를 사용할 수 있다.
KoNLPy를 통해서 사용할 수 있는 형태소 분석기는 다음과 같이 있다.

- Okt(Open Korea Text)
- 메캅(Mecab)
- 코모란(Komoran)
- 한나눔(Hannanum)
- 꼬꼬마(Kkma)

한국어 NLP에서 형태소 분석기를 사용한다는 것은 단어 토큰화가 아니라 정확히는 형태소 단위로 형태소 토큰화를 수행하게 됨을 뜻한다.
여기선 이 중에 Okt Kkma를 통해서 토큰화를 수행해보록 하자.

```python
from konlpy.tag import Okt  
okt=Okt()  

print(okt.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
```

```python
['열심히', '코딩', '한', '당신', ',', '연휴', '에는', '여행', '을', '가봐요']  
```

```python
print(okt.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
```

```python
[('열심히','Adverb'), ('코딩', 'Noun'), ('한', 'Josa'), ('당신', 'Noun'), (',', 'Punctuation'), ('연휴', 'Noun'), ('에는', 'Josa'), ('여행', 'Noun'), ('을', 'Josa'), ('가봐요', 'Verb')]  
```

```python
print(okt.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요")) 
```

```python
['코딩', '당신', '연휴', '여행']  
```

위의 예제는 Okt 형태소 분석기로 토큰화를 시도한 예제이다.

1. morphs:형태소 추출
2. pos: 품사 태깅(Part-of-speech tagging)
3. nouns: 명사 추출

위 예제에서 사용된 각 메소드는 이러한 기능을 가지고 있다.
앞서 언급한 KoNLPy의 형태소 분석기들은 공통적으로 이 메소드를 제공한다.
위 예제에서 형태소 추출과 품사 태깅 메소드의 결과를 보면, 조사를 기본적으로 분리하고 있음을 확인할 수 있다.
그렇기에 한국어 NLP에서 전처리에 형태소 분석기를 사용하는 것은 꽤 유용하다.

이번에는 Kkma 형태소 분석기를 사용하여 같은 문장에 대해 토큰화를 진행해 보자.

```python
from konlpy.tag import Kkma  
kkma=Kkma()  

print(kkma.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
```

```python
['열심히', '코딩', '하', 'ㄴ', '당신', ',', '연휴', '에', '는', '여행', '을', '가보', '아요']  
```

```python
print(kkma.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))  
```

```python
[('열심히','MAG'), ('코딩', 'NNG'), ('하', 'XSV'), ('ㄴ', 'ETD'), ('당신', 'NP'), (',', 'SP'), ('연휴', 'NNG'), ('에', 'JKM'), ('는', 'JX'), ('여행', 'NNG'), ('을', 'JKO'), ('가보', 'VV'), ('아요', 'EFN')]  
```

```python
print(kkma.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))  
```

```python
['코딩', '당신', '연휴', '여행']  
```

앞서 사용한 Okt 형태소 분석기와 결과가 다른 것을 확인할 수 있다.
각 형태소 분석기는 성능과 결과가 다르게 나오기 때문에, 형태소 분석기의 선택은 사용하고자 하는 필요 용도에ㅇ 어떤 형태소 분석기가 가장 적절한지를 판단하고 사용하면 된다.

---

한국어 형태소 분석기 성능 비교 : https://iostream.tistory.com/144
http://www.engear.net/wp/%ED%95%9C%EA%B8%80-%ED%98%95%ED%83%9C%EC%86%8C-%EB%B6%84%EC%84%9D%EA%B8%B0-%EB%B9%84%EA%B5%90/

