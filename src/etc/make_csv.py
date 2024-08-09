import os
import pandas as pd

# 디렉토리 경로 지정
directory_path = '/data/a0c679/wsi_feat/TCGA-ALL-x256-x20-features-from_pan_cancer-from_dino-from_sagemaker-vitb16_dino-epoch=6-bf16/pt_files'

# 디렉토리 내 모든 파일 이름 가져오기
file_names = os.listdir(directory_path)

# 파일 이름을 데이터프레임으로 변환
df = pd.DataFrame(file_names, columns=['slide_id'])

# 결과 확인
print(df)