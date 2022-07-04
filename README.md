# Metaverse-Advertisement-Measure
서강대학교 캡스톤디자인I 산업체 주제 오스리움 : 메타버스 인공지능 광고측정 모델 개발

> [프로젝트 소개 페이지 바로가기](http://cscp2.sogang.ac.kr/CSE4186/index.php/%EC%A1%B0%EA%B9%80%EC%9E%A5%EC%9D%B4)

<br>

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
• Baidu-OCR 구독 만료로 Text Detection 로직은 주석처리된 상태입니다.
  → 과거 기록만 정적 파일(이미지,txt 등)로 남아 웹페이지에 표시됩니다.
  
• DB 연결이 끊어짐에 따라 비디오 및 이미지 파일은 모두 static 폴더 내부에 저장됩니다.

• 서버 배포 방식은 Issue 문서를 참고해 주세요.
```
