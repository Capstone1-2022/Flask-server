## 코드 Clone 받기
- `git clone https://github.com/Capstone1-2022/Flask-server.git`
<br>

## Flask-server 폴더 안에서 가상환경 설치
- `pip3 install virtualenv` (가상환경 설치)
- `virtualenv venv` (가상환경 생성)
- `source venv/bin/activate` (가상환경 활성화)(mac 명령어)
<br>

## 가상환경에 Flask 및 필요한 라이브러리 설치
- `pip3 install flask` (flask 설치)
- `pip3 install -r requirements.txt` (프로젝트에서 사용되는 라이브러리 설치)(시간소요)
<br>

## Flask 앱 실행
- `python3 app.py` (맨 처음에는 시간 소요)
- `http://127.0.0.1:5000/` 접속하여 로컬 서버 사용 가능
<br>

## 참고사항
```
DB 연결이 끊어짐에 따라 비디오 및 이미지 파일은 모두 static 폴더 내부에 저장됩니다.
```
