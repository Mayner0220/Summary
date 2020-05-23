# sText Processing (3)

Source: https://wikidocs.net/21694, https://wikidocs.net/21698, https://wikidocs.net/21693, https://wikidocs.net/21707, https://wikidocs.net/22530, https://wikidocs.net/21703, https://wikidocs.net/31766, https://wikidocs.net/22647, https://wikidocs.net/22592, https://wikidocs.net/33274
NLP에 있어서 Text Processing은 매우 중요한 작업이다.

---

### 4. 불용어(Stopword)

갖고 있는 데이터에서 유의미한 단어 토큰만 선별하기 위해 큰 의미가 없는 단어 토큰을 제거하는 작업은 필요하다.
큰 의미가 없다라는 것은 자주 등장하지만 분석을 하는 것에 있어서 큰 도움이 되지 않는 단어를 의미한다.
예를 들어 I, my, me, over, 조사, 접미사 같은 단어들은 문장에서는 자주 등장하지만 실제 의미분석을 하는데 거의 기여하는 바가 없다.
이러한 단어들을 불용어(stopword)라고 하며, NLTK에서 위와 같은 100여개 이상의 영어 단어들을 불용어 패키지 내에서 미리 정의하고 있다.
물론 개발자가 직접 불용어를 정의할 수도 있다.

---

### 4.1 NLTK에서 불용어 확인하기

```python
from nltk.corpus import stopwords  
stopwords.words('english')[:10]  
```

```python
['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your']
```

stopwords.words('english')는 NLTK가 정의한 영어 불용어 리스트를 리턴한다.
출력 결과가 100개 이상이기 때문에 여기서는 간단히 10개만 확인해보자.
'i', 'me', 'my'와 같은 단어들을 NLTK에서 불용어로 정의하고 있음을 확인 할 수 있다.

---

### 4.2 NLTK를 통해서 불용어 제거하기

```python
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

example = "Family is not an important thing. It's everything."
stop_words = set(stopwords.words('english')) 

word_tokens = word_tokenize(example)

result = []
for w in word_tokens: 
    if w not in stop_words: 
        result.append(w) 

print(word_tokens)
print(result) 
```

```python
['Family', 'is', 'not', 'an', 'important', 'thing', '.', 'It', "'s", 'everything', '.']

['Family', 'important', 'thing', '.', 'It', "'s", 'everything', '.']
```

위 코드는 "Family is not an important thing. It's everything."라는 임의의 문장을 정의하고, NLTK가 정의하고 있는 불용어를 제외한 결과를 출력하고 있다.
'is', 'not', 'an'과 같은 단어들이 제거됬음을 알 수 있다.

---

### 4.3 한국어에서 불용어 제거하기

위 코드는 불용어를 제거하는 방법으로는 간단하게는 토큰화 후에 조사, 접속사등을 제거하는 방법이 있다.
하지만 불용어를 제거하려고 하다보면 조사나 접속사와 같은 단어들뿐만 아니라 명사, 형요사와 같은 단어들 중에서 불용어로서 제거하고 싶은 단어들이 생기기도 한다.
결국에는 사용자가 직접 불용어 사전을 만들게 되는 경우가 많다.
이번에는 직접 불용어를 정의해보며, 주어진 문장으로 부터 직접 정의한 불용어 사전을 참고로 불용어를 제거해보자.

```python
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

example = "고기를 아무렇게나 구우려고 하면 안 돼. 고기라고 다 같은 게 아니거든. 예컨대 삼겹살을 구울 때는 중요한 게 있지."
stop_words = "아무거나 아무렇게나 어찌하든지 같다 비슷하다 예컨대 이럴정도로 하면 아니거든"
# 위의 불용어는 명사가 아닌 단어 중에서 저자가 임의로 선정한 것으로 실제 의미있는 선정 기준이 아님
stop_words=stop_words.split(' ')
word_tokens = word_tokenize(example)

result = [] 
for w in word_tokens: 
    if w not in stop_words: 
        result.append(w) 
# 위의 4줄은 아래의 한 줄로 대체 가능
# result=[word for word in word_tokens if not word in stop_words]

print(word_tokens) 
print(result)
```

```python
['고기를', '아무렇게나', '구우려고', '하면', '안', '돼', '.', '고기라고', '다', '같은', '게', '아니거든', '.', '예컨대', '삼겹살을', '구울', '때는', '중요한', '게', '있지', '.']

['고기를', '구우려고', '안', '돼', '.', '고기라고', '다', '같은', '게', '.', '삼겹살을', '구울', '때는', '중요한', '게', '있지', '.']
```

아래의 링크는 보편적으로 선택할 수 잇는 한국어 불용어 리스트를 보여준다.
(단, 절대적인 기준은 아니다.)
*링크: https://www.ranks.nl/stopwords/korean

한국어 불용어를 제거하는 더  좋은 방법은 코드 내에서 직접 정의하지 않고 txt파일이나 csv파일로 불용어를 정리하고, 이를 불러와서 사용하는 방법이다.

---

### 5. 정규 표현식(Regular Expression)

텍스트 데이터를 전처리하다보면, 정규 표현식은 아주 유용한 도구로서 사용된다.
이번에는 파이썬에서 지원하고 있는 정규 표현식 모듈 re의 사용 방법과 NLTK를 통한 정규 표현식을 이용한 토큰화에 대해서 알아보도록 하자.

---

### 5.1 **정규 표현식 문법과 모듈 함수**

파이썬에서는 정규 표현식 모듈 re를 지원하므로, 이를 이용하면 특정 규칙이 있는 텍스트 데이터를 빠르게 정제할 수 있다.

1. 정규 표현식 문법

   정규 표현식을 위해 사용되는 문법 중 특수 문자들은 아래와 같다.

|    특수문자    | 설명                                                         |
| :------------: | ------------------------------------------------------------ |
|       .        | 한 개의 임의 문자를 나태냅니다. (줄바꿈 문자인 \는 제외)     |
|       ?        | 앞의 문자가 존재할 수도 있고, 존재하지 않을 수도 있다. (문자가 0개/1개) |
|       *        | 앞의 문자가 무한개로 존재할 수 있고, 존재하지 않을 수도 있다. (문자가 0개 이상) |
|       +        | 앞의 문자가 최소 한 개 이상 존재한다. (문자가 1개 이상)      |
|       ^        | 뒤의 문자로 문자열이 시작된다.                               |
|       $        | 앞의 문자로 문자열이 끝난다.                                 |
|     {숫자}     | 숫자만큼 반복한다.                                           |
| {숫자1, 숫자2} | 숫자1 이상 숫자2 이하만큼 반복한다. ?, *, +를 이것으로 대체할 수 있다. |
|    {숫자, }    | 숫자 이상만큼 반복한다.                                      |
|      [ ]       | 대괄호 안의 문자들 중 한 개의 문자와 매치한다. [amk]라고 한다면 a 또는 m 또는 k 중 하나라도 존재하면 매치를 의미한다. [a-z]와 같이 범위를 지정할 수 있다. [a-zA-Z]는 알파벳 전체를 의미하는 범위이며, 문자열에 알파벳이 존재하면 매치를 의미한다. |
|    [^숫자]     | 해당 문자를 제외한 문자를 매치한다.                          |
|       \|       | A\|B와 같이 쓰이며, A 또는 B의 의미를 가진다.                |

정규 표현식 문법에는 역 슬래쉬(\)를 이용하여 자주 쓰이는 문자 규칙이 있다.

| 문자 규칙 | 설명                                                         |
| :-------: | ------------------------------------------------------------ |
|    \\     | 역 슬래쉬 문자 자체를 의미한다.                              |
|    \d     | 모든 숫자를 의미한다. [0-9]와 의미가 동일하다.               |
|    \D     | 숫자를 제외한 모든 문자를 의미한다. [^0-9]와 의마가 동일하다. |
|    \s     | 공백을 의미한다. [ \t\n\r\f\v]와 의미가 동일하다.            |
|    \S     | 공백을 제외한 문자를 의미한다. [^ \t\n\r\f\v]와 의미가 동일하다. |
|    \w     | 문자 또는 숫자를 의미한다. [a-zA-Z0-9]와 의미가 동일하다.    |
|    \W     | 문자 또는 숫자가 아닌 문자를 의미한다. [^a-zA-Z0-9]와 의미가 동일하다. |

2. 정규표현식 모듈 함수
   정규표현식 모듈에서 지원하는 함수는 이와 같다.

|   모듈 함수   | 설명                                                         |
| :-----------: | ------------------------------------------------------------ |
| re.compile()  | 정규표현식을 컴파일하는 함수이다. 다시 말해, 파이썬에게 전해주는 역할이다. 찾고자 하는 패턴이 빈번한 경우에는 미리 컴파일해놓고 사용하면 속도와 편의성면에서 유리하다. |
|  re.search()  | 문자열 전체에 대해서 정규표현식과 매치되는지를 검색한다.     |
|  re.match()   | 문자열의 처음이 정규표현식과 매치되는지를 검색한다.          |
|  re.split()   | 정규 표현식을 기준으로 문자열을 분리하여 리스트로 반환한다.  |
| re.findall()  | 문자열에서 정규 표현식과 매치되는 모든 경우의 문자열을 찾아서 리스트로 리턴한다. 만약, 매치되는 문자열이 없다면 빈 리스트가 리턴된다. |
| re.finditer() | 문자열에서 정규 표현식과 매치되는 모든 경우의 문자열에 대한 이터레이터 객체를 리턴한다. |
|   re.sub()    | 문자열에서 정규 표현식과 일치하는 부분에 대해서 다른 문자열로 대체한다. |

앞으로 진행될 실습에서는 re.compile()에 정규 표현식을 컴파일하고, re.search()를 통해서 해당 정규 표현식이 입력 테스트와 매치되는지를 확인하면서 각 정규 표현식에 대해서 이해해보도록 하자.
re.search() 함수는 매치된다면 Match Object를 리턴하고, 매치되지 않으면 아무런 값도 출력하지 않는다.

---

### 5.2 정규 표현식 실습

1. .기호
   .은 한 개의 임의의 문자를 나타낸다.
   예를 들어 정규 표현식이 a.c이라고 가정하자.
   a와 c 사이에는 어떤 1개의 문자라도 올 수 있다.
   즉, akc, azc, avc, a5c, a!c와 같은 형태는 모두 a.c의 정규 표현식과 매치된다.

   ```python
   import re
   r=re.compile("a.c")
   r.search("kkk") # 아무런 결과도 출력되지 않는다.
   r.search("abc")
   <_sre.SRE_Match object; span=(0, 3), match='abc'>  
   ```

   위의 코드는 search의 입력으로 들어오는 문자열에 정규표현식 패턴 a.c이 존재하는지를 확인하는 코드이다.
   (.)은 어떤 문자로도 인식될 수 있기 때문에 abc라는 문자열은 a.c라는 정규표현식 패턴으로 매치된다.

2. ?기호
   ?는 ? 앞의 문자가 존재할 수도 있고, 존재하지 않을 수도 있는 경우를 나타냅니다.
   예를 들어서 정규 표현식이 a?c라고 가정하자.
   이 경우 정규 표현식에서의 b는 있다고 취급할 수도 있고, 없다고 취급할 수도 있다.
   즉, abc와 ac 모두 매치할 수 있다.

   ```python
   import re
   r=re.compile("ab?c")
   r.search("abbc") # 아무런 결과도 출력되지 않는다.
   r.search("abc")
   ```
   
   ```python
<_sre.SRE_Match object; span=(0, 3), match='abc'>  
   ```

   b가 있는 것으로 판단하여 abc를 매치하는 것을 볼 수 있다.
   
   ```python
   r.search("ac")
   ```

   ```python
<_sre.SRE_Match object; span=(0, 2), match='ac'>  
   ```
   
   b가 없는 것으로 판단하여 ac를 매치하는 것을 볼 수 있다.
   
3. *기호
   *은 바로 앞의 문자가 0개 이상일 경우를 나타낸다.
   앞의 문자는 존재하지 않을 수도 있으며, 또는 여러 개일 수도 있다.
   예를 들어서 정규 표현식이 abc라고 가정하자.
   그렇다면 ac, abc, abbc, abbbc 등과 매치할 수 있으며 b의 갯수는 무수히 많아도 상관 없다.

   ```python
   import re
   r=re.compile("ab*c")
   
   r.search("a") # 아무런 결과도 출력되지 않는다.
   
   r.search("ac")
   ```
   
   ```python
   <_sre.SRE_Match object; span=(0, 2), match='ac'>  
   ```
   
   ```python
   r.search("abc") 
   ```
```
   
   ```python
   <_sre.SRE_Match object; span=(0, 3), match='abc'> 
```

   ```python
   r.search("abbbbc") 
   ```

   ```python
   <_sre.SRE_Match object; span=(0, 6), match='abbbbc'> 
   ```

4. +기호
   +는 *와 유사하다.
   하지만 다른 점은 앞의 문자가 최소 1개 이상이어야 한다는 점이다.
   예를 들어서 정규 표현식이 ab+c라고 한다면, ac는 매치되지 않는다.
   하지만  abc, abbc, abbbc 등과 매치할 수 있으며 b의 갯수는 무수히 많을 수 있다.

   ```python
   import re
   r=re.compile("ab+c")
   
   r.search("ac") # 아무런 결과도 출력되지 않는다.
   
   r.search("abc") 
   ```

   ```python
   <_sre.SRE_Match object; span=(0, 3), match='abc'>   
   ```

   ```python
   r.search("abbbbc") 
   ```

   ```python
   <_sre.SRE_Match object; span=(0, 6), match='abbbbc'>  
   ```

5. ^기호
   ^는 시작되는 글자를 지정한다.
   가령 정규표현식이 ^a라면 a로 시작되는 문자열만을 찾아낸다.

   ```python
   import re
   r=re.compile("^a")
   
   r.search("bbc") # 아무런 결과도 출력되지 않는다.
   
   r.search("ab")
   ```

   ```python
   <_sre.SRE_Match object; span=(0, 1), match='a'>  
   ```

   bbc는 a로 시작되지 않지만, ab는 a로 시작되기 때문에 매치되었다.

6. {숫자} 기호

   문자에 해당 기호를 붙이면, 해당 문자를 숫자만큼 반복한 것을 나태낸다.

   예를 들어서 정규 표현식이 ab{2}c라면 a와 c 사이에 존재하면서 b가 2개인 문자열에 대해서 매치한다.

   ```python
   import re
   r=re.compile("ab{2}c")
   
   r.search("ac") # 아무런 결과도 출력되지 않는다.
   
   r.search("abc") # 아무런 결과도 출력되지 않는다.
   ```

   ```python
   r.search("abbc")
   ```

   ```python
   <_sre.SRE_Match object; span=(0, 4), match='abbc'>
   ```

   ```python
   r.search("abbbbbc") # 아무런 결과도 출력되지 않는다.
   ```

7. {숫자1, 숫자2} 기호
   문자에 해당 기호를 붙이면, 해당 문자를 숫자1 이상 숫자2 이하만큼 반복한다.
   예를 들어서 정규 표현식이 ab{2, 8}c라면 a와 c 사이에 b가 존재하면서 b는 2개 이상 8개 이하인 문자열에 대해서 매치한다.

   ```python
   import re
   r=re.compile("ab{2,8}c")
   
   r.search("ac") # 아무런 결과도 출력되지 않는다.
   
   r.search("ac") # 아무런 결과도 출력되지 않는다.
   
   r.search("abc") # 아무런 결과도 출력되지 않는다.
   
   r.search("abbc")
   ```

   ```python
   <_sre.SRE_Match object; span=(0, 4), match='abbc'>
   ```

   ```python
   r.search("abbbbbbbbc")
   ```

   ```python
   <_sre.SRE_Match object; span=(0, 10), match='abbbbbbbbc'>
   ```

   ```python
   r.search("abbbbbbbbbc") # 아무런 결과도 출력되지 않는다.
   ```

8. {숫자, } 기호
   문자에 해당 기호를 붙이면 해당 문자를 숫자 이상 만큼 반복한다.
   예를 들어서 정규 표현식이 a{2, }bc라면 뒤에 bc가 붙으면서 a의 갯수가 2개 이상인 경우인 문자열과 매치한다.
   또한 만약 {0,}을 쓴다면 *와 동일한 의미가 되며, {1,}을 쓴다면 +와 동일한 의미가 된다.

   ```python
   import re
   r=re.compile("a{2,}bc")
   
   r.search("bc") # 아무런 결과도 출력되지 않는다.
   
   r.search("aa") # 아무런 결과도 출력되지 않는다.
   
   r.search("aabc")
   ```

   ```python
   <_sre.SRE_Match object; span=(0, 4), match='aabc'>
   ```

   ```python
   r.search("aaaaaaaabc")
   ```

   ```python
   <_sre.SRE_Match object; span=(0, 10), match='aaaaaaaabc'> 
   ```

9. [ ] 기호
   [ ]안에 문자들을 넣으면 그 문자들 중 한 개의 문자와 매치라는 의미를 가진다.
   예를 들어서 정규 표현식이 [abc]라면, a 또는 b 또는 c가 들어가 있는 문자열과 매치된다.
   범위를 지정하는 것도 가능하다.
   [a-zA-Z]는 알파벳이 전부를 의미하며, [0-9]는 숫자 전부를 의미한다.

   ```python
   import re
   r=re.compile("[abc]") # [abc]는 [a-c]와 같다.
   
   r.search("zzz") # 아무런 결과도 출력되지 않는다.
   
   r.search("a")
   ```

   ```python
   <_sre.SRE_Match object; span=(0, 1), match='a'> 
   ```

   ```python
   r.search("aaaaaaa")                                                                               
   ```

   ```python
   <_sre.SRE_Match object; span=(0, 1), match='a'> 
   ```

   ```python
   r.search("baac")      
   ```

   ```python
   <_sre.SRE_Match object; span=(0, 1), match='b'>
   ```

   이번에는 알파벳 소문자에 대해서만 범위 지정하여 정규 표현식을 만들어보고 문자열과 매치해보도록 하자.

   ```python
   import re
   r=re.compile("[a-z]")
   
   r.search("AAA") # 아무런 결과도 출력되지 않는다.
   ```

   ```python
   r.search("aBC")
   ```

   ```python
   <_sre.SRE_Match object; span=(0, 1), match='a'>
   ```

   ```python
   r.search("111") # 아무런 결과도 출력되지 않는다.
   ```

10. [^문자] 기호
    [^문자]는 위에서 설명한 ^와 ㅇ완전히 다른 의미로 쓰인다.
    여기서 ^ 기호 뒤에 붙은 문자들을 제외한 모든 문자를 매치하는 역할을 수행한다.
    예를 들ㅇ서 [^abc]라는 정규 표현식이 있다면, a 또는 b 또는 c가 들어간 문자열을 제외한 모든 문자열을 매치한다.

    ```python
    import re
    r=re.compile("[^abc]")
    
    r.search("a") # 아무런 결과도 출력되지 않는다.
    
    r.search("ab") # 아무런 결과도 출력되지 않는다.
    
    r.search("b") # 아무런 결과도 출력되지 않는다.
    
    r.search("d")
    ```

    ```python
    <_sre.SRE_Match object; span=(0, 1), match='d'> 
    ```

    ```python
    r.search("1")                                                                                     
    ```

    ```python
    <_sre.SRE_Match object; span=(0, 1), match='1'> 
    ```

---

### 5.3 정규 표현식 모듈 함수 예제

지금까지 정규 표현식 문법에 대한 이해를 정규 표현식 모듈 함수 중에서 re.compile()과 re.search를 사용해봤다.
이번에는 다른 정규 표현식 모듈 함수에 대해서도 직접 실습을 진행해보자.

1. re.match() 와 re.search()의 차이
   search()가 정규 표현식 전체에 대해서 문자열이 매치하는지를 본다면, match()는 문자열의 첫 부분부터 정규 표현식과 매치하는지를 확인한다.
   문자열 중간에 찾을 패턴이 있다고 하더라도, match 함수는 문자열의 시작에서 일치하지 않으면 찾지 않는다.

   ```python
   import re
   r=re.compile("ab.")
   
   r.search("kkkabc")   
   ```
   
   ```python
   <_sre.SRE_Match object; span=(3, 6), match='abc'>   
   ```
   
   ```python
r.match("kkkabc")  #아무런 결과도 출력되지 않는다.
   ```
   
   ```python
   r.match("abckkk")  
   ```
```
   
   ```python
   <_sre.SRE_Match object; span=(0, 3), match='abc'> 
```

   위의 경우 정규 표현식이 ab. 이기에, ab 다음에는 어떤 한 글자가 존재할 수 있다는 패턴을 의미한다.
   search 모듈 함수에 kkkabc라는 문자열을 넣어 매치되는지 확인한다면 abc라는 문자열에서 매치되어 Match object를 리턴한다.
   하지만 match 모듈 함수의 경우 앞 부분이 ab.와 매치되지 않기 때문에, 아무런 결과도 출력되지 않는다.
   하지만 반대로 abckkk로 매치를 시도해보면,  시작 부분에서 패턴과 매치되었기 때문에 정상적으로 Match object를 리턴한다.

2. re.split()
   split() 함수는 입력된 정규 표현식을 기준으로 문자열들을 분리하여 리스트로 리턴한다.
   NLP에 있어서 가장 많이 사용되는 정규 표현식 함수 중 하나인데, 토큰화에 유용하게 쓰일 수 있기 때문이다.

   ```python
   import re
   text="사과 딸기 수박 메론 바나나"
   
   re.split(" ",text)
   ```
   
   ```python
['사과', '딸기', '수박', '메론', '바나나']  
   ```

   위의 예제의 경우 입력 텍스트로 부터 공백을 기준으로 문자열 분리를 수행하였고, 결과로서 리스트를 리턴하는 모습을 볼 수 있다.
   
   ```python
   import re
   text="""사과
   딸기
   수박
   메론
   바나나"""
   
   re.split("\n",text)
   ```

   ```python
['사과', '딸기', '수박', '메론', '바나나']  
   ```
   
   이와 유사하게 줄바꿈이나 다른 정규 표현식을 기준으로 텍스트를 분리할 수도 있다.
   
3. re.findall()
   findall() 함수는 정규 표현식과 매치되는 모든 문자열들을 리스트로 리턴한다.
   단, 매치되는 문자열이 없다면 빈 리스트를 리턴한다.

   ```python
   import re
   text="이름 : 김철수
   전화번호 : 010 - 1234 - 1234
   나이 : 30
   성별 : 남"""  
   
   re.findall("\d+",text)
   ```
   
   ```python
['010', '1234', '1234', '30']
   ```
   

정규 표현식으로 숫자를 입력하면 전체 텍스트로부터 숫자만 찾아내서 리스트로 리턴하는 것을 볼 수 있다.
   하지만 만약 입력 테스트에 숫자가 없다면 빈 리스트를 리턴한다.

   ```python
   re.findall("\d+", "문자열입니다.")
   ```

   ```python
   [] # 빈 리스트를 리턴한다.
   ```

4. re.sub()
   sub() 함수는 정규 표현식 패턴과 일치하는 문자열을 찾아 다른 문자열로 대체할 수 있다.

   ```python
   import re
   text="Regular expression : A regular expression, regex or regexp[1] (sometimes called a rational expression)[2][3] is, in theoretical computer science and formal language theory, a sequence of characters that define a search pattern."
   
   re.sub('[^a-zA-Z]',' ',text)
   ```

   ```python
   'Regular expression   A regular expression  regex or regexp     sometimes called a rational expression        is  in theoretical computer science and formal language theory  a sequence of characters that define a search pattern '  
   ```

---

### 5.5 정규 표현식 텍스트 전처리 예제

```python
import re  
text = """100 John    PROF
101 James   STUD
102 Mac   STUD"""  

re.split('\s+', text)  
```

'\s+'는 공백을 찾아내는 정규표현식이다.
뒤에 붙는 +는 최소 1개 이상의 패턴을 찾아낸다는 의미이다.
s는 공백을 의미하기에 최소 1개 이상의 공백인 패턴을 찾아낸다,
입력으로 테이블 형식의 데이터를 텍스트에 저장했다,
각 데이터가 공백으로 구분되어 있다.
split은 주어진 정규표현식을 기준으로 분리하기에, 결과는 아래와 같다.

```python
['100', 'John', 'PROF', '101', 'James', 'STUD', '102', 'Mac', 'STUD']
```

이제 \d는 숫자에 해당되는 정규표현식이다.
+를 붙였으므로 최소 1개 이상의 숫자에 해당하는 값을 의미한다.
findall은 해당 정규 표현식에 일치하는 값을 찾아내는 메소드이다.

```python
['100', '101', '102]
```

해당 코드의 결과는 위와 같다.
이번에는 텍스트로부터 대문자인 행의 값만 가져오고 싶다고 가정하자.
이 경우에는 정규 표현식에 대문자를 기준으로 매치시키면 된다.
하지만 정규 표현식에 대문자라는 기준만을 넣을 경우에는 문자열을 가져오는 것이 아니라 모든 대문자 각각을 갖고오게 된다.

```python
re.findall('[A-Z]',text)
```

```python
['J', 'P', 'R', 'O', 'F', 'J', 'S', 'T', 'U', 'D', 'M', 'S', 'T', 'U', 'D']
```

이는 우리가 원하는 결과가 아니다.
이 경우, 여러가지 방법이 있겠지만 대문자가 연속적으로 4번 등장하는 경우로 조건을 추가해보자.

```python
re.findall('[A-Z]{4}',text)  
```

```python
['PROF', 'STUD', 'STUD']
```

대문자로 구성된 문자열들을 제대로 가져오는 것을 볼 수 있다.
이름의 경우에는 대문자와 소문자가 섞여있는 상황이다.
이름에 대한 행의 값을 갖고오고 싶다면 처음에 대문자가 등장하고, 그 후에 소문자가 등장하는 경우에 매치하게 한다.

```python
re.findall('[A-Z][a-z]+',text)
```

```python
['John', 'James', 'Mac'] 
```

```python
import re

letters_only = re.sub('[^a-zA-Z]', ' ', text)
```

---

### 5.6 정규 표현식을 이용한 토큰화

NLTK에서는 정규 표현식을 사용해서 단어 토큰화를 수행하는 RegexpTokenizer를 지원한다.
RegexpTokenizer()에서 괄호 안에 원하는 정규 표현식을 넣어서 토큰화를 수행하는 것이다.

```python
import nltk
from nltk.tokenize import RegexpTokenizer
tokenizer=RegexpTokenizer("[\w]+")

print(tokenizer.tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop"))
```

```python
['Don', 't', 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', 'Mr', 'Jone', 's', 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop'] 
```

tokenizer=RegexpTokenizer("[\w]+")에서 \+는 문자 또는 숫자가 1개 이상인 경우를 인식하는 코드이다.
그렇기에, 이 코드는 문장에서 구두점을 제외하고, 단어들만 가지고 토큰화를 수행한다.

RexpTokenizer()에서 괄호 안에 토큰으로 원하는 정규 표현식을 넣어서 사용한다고 언급했다.
그런데 괄호 안에 토큰을 나누기 위한 기준을 입력할 수 있다.
이번에는 공백을 기준으로 문장을 토큰화해보도록 하자.

```python
import nltk
from nltk.tokenize import RegexpTokenizer
tokenizer=RegexpTokenizer("[\s]+", gaps=True)

print(tokenizer.tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop"))
```

```python
["Don't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name,', 'Mr.', "Jone's", 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop']
```

위 코드에서 gaps=true는 해당 정규 표현식을 토큰으로 나누기 위한 기준으로 사용한다는 의미이다.
만약 gaps=True라는 부분을 기재하지 않는다면, 토큰화의 결과는 공백들만 나오게 된다.
이번에는 위의 예제와는 달리 아포스트로피나 온점을 제외하지 않고, 토큰화가 수행된 것을 확인 할 수 있다.