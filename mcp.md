# MCP 실습

## 실습에 필요한 함수 설정

### 날씨 API

1) [openweathermap](https://home.openweathermap.org/users/sign_in)에 접속합니다.
2) 메뉴에서 [API를 선택한 후 스크롤하여 [Current Weather Data]을 Subscribe 합니다.
3) [Free] Plan에서 [Get API key]을 선택하여 key를 복사합니다.
4) config.json 파일을 생성해서 아래와 같이 입력합니다.

아래와 같이 입력합니다.

```java
{
    "WEATHER_API_KEY": "fbd----",
    "TAVILY_API_KEY": "tvly-abcd"
}
```


## Default

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


