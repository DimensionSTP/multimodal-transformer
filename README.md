# Multimodal Transformer

## 실험 환경 : google colab pro

### 실험 세팅

* 코드 수정 필요시 **colab-ssh** 연결로 수정함[(참고)](:https://blog.naver.com/PostView.nhn?blogId=ys10mjh&logNo=222328257839&parentCategoryNo=&categoryNo=29&viewDate=&isShowPopularPosts=true&from=search)

* git clone이 아닌 zip file download로 **구글 드라이브**에 저장

* 코드의 colab notebook에 맞춰 google drive에 폴더 생성 필요


### Single Modal Training

* only_audio...ipynb와 only_text...ipynb로 각각 single modal의 checkpoint get

* ETRI_voting.ipynb로 single modality F1 score, weighted sum F1 score 확인

### Multi Modal Training

* multi_modal.ipynb로 cross-modal transformer training, 해당 노트북에 hydra config 사용을 위해 데이터 경로를 colab의 절대 경로로 바꾸는 과정 포함

### 결과 저장
* 구글 드라이브 메인에 해당 코드 zip file, ETRI 폴더에 KEMDy19 데이터, gdrive/ETRI/backup/ 및 gdrive/ETRI/npy/에 각각 single modal model checkpoint, pred 값