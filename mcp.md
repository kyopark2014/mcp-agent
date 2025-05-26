# MCP 실습

## 실습에 필요한 Key 설정

### 날씨 API

1) [openweathermap](https://home.openweathermap.org/users/sign_in)에 접속합니다.
2) 메뉴에서 [API를 선택한 후 스크롤하여 [Current Weather Data]을 Subscribe 합니다.
3) [Free] Plan에서 [Get API key]을 선택하여 key를 복사합니다.

### Tavily Search

1) [Tavily.com](https://www.tavily.com/)에 접속합니다.
2) API key를 복사합니다.

### Key의 보관

config.json 파일을 생성해서 아래와 같이 입력합니다.

아래와 같이 입력합니다.

```java
{
    "WEATHER_API_KEY": "fbd----",
    "TAVILY_API_KEY": "tvly-abcd"
}
```


## Tool의 사용

### Default

"현재 시간은?"으로 입력 후 결과를 확인합니다.

![image](https://github.com/user-attachments/assets/cb6ab7e2-9578-45c7-8e51-cd5dbd1a0719)

이 동작을 위한 함수는 아래와 같습니다.

```python
def get_current_time(format: str=f"%Y-%m-%d %H:%M:%S")->str:
    """Returns the current date and time in the specified format"""
    
    format = format.replace('\'','')
    timestr = datetime.datetime.now(timezone('Asia/Seoul')).strftime(format)
    logger.info(f"timestr: {timestr}")
    
    return timestr
```

"여행하면서 읽기 좋은 책은?"이라고 입력후 결과를 확인합니다.

![image](https://github.com/user-attachments/assets/d8457806-32a2-4fa3-addb-b15d2c0c0482)

"서울 날씨는"이라고 입력후 결과를 확인합니다.

![image](https://github.com/user-attachments/assets/7c4adeae-77a0-4630-92b6-a14eaa69b5e4)

아래와 같이 tool을 설정합니다.

![image](https://github.com/user-attachments/assets/1bae6af5-f93c-412e-86c4-b4bd61435794)

"네이버 주식 현황은?"이라고 입력합니다. yahoo stock으로 실패했습니다.

![image](https://github.com/user-attachments/assets/b8b0c28e-f231-4c9e-803b-7a111eac4b7a)

### Code Interpreter

"strawberry의 r의 갯수는?"라고 입력 후에 결과를 확인합니다.

![image](https://github.com/user-attachments/assets/0bb90de0-b8dc-4651-8384-051114f5770f)

"가우시안 분포를 그래프로 그려주세요."라고 입력하면 아래와 같이 repl_drawer가 활용되어 생성된 코드로 그림을 그립니다. 

![image](https://github.com/user-attachments/assets/97ae0842-02e5-49b8-ab77-85e031e17972)

결과는 아래와 같습니다.

![image](https://github.com/user-attachments/assets/f1e995ef-2e0a-499a-a1de-e4d6655f5355)

### AWS Documentation

"aws의 secret key를 안전하게 보관하는 정책은?"와 같이 입력후 결과를 확인합니다.

![image](https://github.com/user-attachments/assets/0885e918-9e66-458b-9aef-4a6b78ad321e)

### AWS CLI

"내가 가지고 있는 모든 aws 리소스는?"와 같이 입력후 결과를 확인합니다.

![image](https://github.com/user-attachments/assets/dc1d25ed-d431-4ac1-80a8-39d968629304)

### AWS CloudWatch

"내 cloudwatch 로그 현황은?"와 같이 입력후 결과를 확인합니다.

![image](https://github.com/user-attachments/assets/061ccac7-f886-4a6f-9997-45656aec68b0)

### AWS Storage

"내 스토리지 현황은?"와 같이 입력후 결과를 확인합니다.

![image](https://github.com/user-attachments/assets/a96ad8a1-45f2-4997-bf04-9fb022971dea)

### AWS Diagram

"api gateway - cloudfront - s3로 구성되는 cloud architecture 그려주세요."와 같이 입력후 결과를 확인합니다.

![image](https://github.com/user-attachments/assets/2cb691f6-e1e5-4da4-9fd6-f582aa936c8e)

이때의 결과는 아래와 같습니다

![image](https://github.com/user-attachments/assets/858382ad-e53e-4e7d-b00e-f347545a09d8)

### Tavily 인터넷 검색

"강남역 맛집은?"와 같이 입력후 결과를 확인합니다.

![image](https://github.com/user-attachments/assets/c730dd74-0896-463f-9891-38c0856dbec4)

### Arxiv 

"reasoning은 무엇인가요?"와 같이 입력후 결과를 확인합니다.

![image](https://github.com/user-attachments/assets/a5a9786a-5b76-49fc-8a9e-d4478a8fb93b)

### Wikipedia

"k-pop의 특징은?"와 같이 입력후 결과를 확인합니다.

![image](https://github.com/user-attachments/assets/c9f41b01-c2d1-404b-91ca-db494b28df4e)

### Filesystem

"내 폴더의 파일 리스트?"와 같이 입력후 결과를 확인합니다.

![image](https://github.com/user-attachments/assets/66239f7d-ce1b-439e-846a-b55fc8695b96)

### Playwright

"https://github.com/kyopark2014/mcp 의 내용을 요약하세요."와 같이 입력후 결과를 확인합니다.

![image](https://github.com/user-attachments/assets/fee02f29-08fc-476c-a62e-fc2b412a565f)
