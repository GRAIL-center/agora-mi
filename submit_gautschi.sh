#!/bin/bash
#SBATCH -A your_allocation         # <-- TODO: 사용할 할당(Allocation) 이름으로 변경하세요!
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1          # A100 GPU 1장 요청
#SBATCH --mem=120G                 # 혹시 모를 로딩 OOM 방지를 위해 80G -> 120G로 넉넉히 상향
#SBATCH -t 24:00:00                # 최대 24시간 가동
#SBATCH -J aiforge_4models         # 작업 이름
#SBATCH -o logs/slurm-%j.out       # 표준 출력 로그
#SBATCH -e logs/slurm-%j.err       # 에러 출력 로그

echo "=========================================================="
echo "    Starting AI Forge Pipeline on Gautschi A100 Cluster   "
echo "=========================================================="
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "=========================================================="

# 환경 설정 및 모듈 로딩 (학교 서버 환경에 따라 주석 해제하여 사용)
# module load anaconda3/2023.09-0
# module load cuda/12.1

# 가상환경 venv 생성 및 패키지 설치 (자동화)
if [ ! -d "venv" ]; then
    echo "가상환경(venv)을 새로 생성합니다..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    echo "의존성 패키지 설치 중..."
    pip install -r requirements.txt
    pip install git+https://github.com/aengusl/circuit-tracer.git --force-reinstall
else
    echo "기존 가상환경(venv)을 로드합니다..."
    source venv/bin/activate
fi

# HuggingFace 인증 체크 (로그인 노드에서 미리 1번만 huggingface-cli login 수행 필요)
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN 환경 변수가 설정되지 않았습니다. 로그인 노드에서 미리 huggingface-cli login을 완료했기를 바랍니다."
fi

# 파이프라인 필수 환경 변수
export PYTHONPATH=$PYTHONPATH:$(pwd)/src:$(pwd)
export TF_ENABLE_ONEDNN_OPTS=0

# 디렉토리 보장
mkdir -p logs results

echo "모든 패키지 및 환경 설정 완료. 4개 모델 순차 파이프라인 시작!"
bash run_all_models.sh

echo "=========================================================="
echo "Job Finished Successfully at $(date)."
echo "=========================================================="
