# Flask-server

## 서비스 웹페이지 URL
```
http://3.35.218.28:5000/
```

## EC2 서버 접속 방법
- pem 키파일 있는 경로에서 아래의 명령어로 접속 가능
- 000-000-000-000 부분에는 ip 주소 입력
```
ssh -i "키파일이름" ubuntu@ec2-000-000-000-000.compute-1.amazonaws.com
```

## Flask 웹서버 실행 방법
> Flask 앱을 실행하기 위한 라이브러리 설치하기
```
sudo apt update
sudo apt install openjdk-8-jre
sudo apt install openjdk-8-jdk
sudo apt install python3-pip
```
> github 레포 또는 vscode를 이용하여 Flask 패키지를 서버에 올리기

> 개인 프로젝트에 필요한 라이브러리 설치하기
```
pip3 install -r requirements.txt
```
> Flask `app.py` 파일 내용 수정하기
- AWS 서버의 연결 소스를 0.0.0.0/0으로 하기 때문에 host를 '0.0.0.0'으로 해줘야 함.
- host를 입력하지 않으면 127.0.0.1과 같이 다른 host로 연결을 시도하기 때문에 필수로 입력해주자.
```
if __name__ == '__main__':
        app.run(host='0.0.0.0')
```
> Flask 앱 백그라운드에서 실행하기
```
nohup python3 -u app.py &
```
> 앱이 잘 실행되고 있는지 확인할 수 있는 명령어들
```
ps -aux|grep python3 (실행중인 프로세스 확인 명령어)
tail -f nohup.out (로그 확인 명령어)
```

## VSCode에서 EC2 접속하는 방법
- [링크](https://deepmal.tistory.com/8) 참고하여 진행
- `ForwardAgent`는 작성할 필요 없음
- 홈 디렉토리의 .ssh 폴더 내부에 pem 키파일을 위치시킬 것
- User 는 반드시 `ubuntu` 라고 작성해야 함
<br>
