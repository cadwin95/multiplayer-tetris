import numpy as np
import matplotlib.pyplot as plt
import os

def view_training_data():
    """학습 데이터 확인"""
    # 데이터 로드
    print("\n데이터 로드 중...")
    X = np.load('./data/X_train.npy')
    y = np.load('./data/y_train.npy')
    scores = np.load('./data/scores.npy')
    
    # 기본 정보 출력
    print("\n=== 데이터 기본 정보 ===")
    print(f"상태(X) 형태: {X.shape}")
    print(f"행동(y) 형태: {y.shape}")
    print(f"점수 개수: {len(scores)}")
    
    # 행동 분포 확인
    action_counts = np.bincount(y)
    action_names = ['Left', 'Right', 'Rotate', 'Down', 'HardDrop']
    
    print("\n=== 행동 분포 ===")
    for action, count in enumerate(action_counts):
        percentage = (count / len(y)) * 100
        print(f"{action_names[action]}: {count} ({percentage:.2f}%)")
    
    # 점수 통계
    print("\n=== 점수 통계 ===")
    print(f"평균 점수: {np.mean(scores):.2f}")
    print(f"최고 점수: {np.max(scores):.2f}")
    print(f"최저 점수: {np.min(scores):.2f}")
    print(f"표준 편차: {np.std(scores):.2f}")
    
    # 시각화
    plt.figure(figsize=(15, 5))
    
    # 1. 행동 분포
    plt.subplot(131)
    plt.bar(action_names, action_counts)
    plt.title('Action Distribution')
    plt.xticks(rotation=45)
    
    # 2. 점수 분포
    plt.subplot(132)
    plt.hist(scores, bins=30)
    plt.title('Score Distribution')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    
    # 3. 점수 변화
    plt.subplot(133)
    plt.plot(scores)
    plt.title('Score Progression')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    plt.tight_layout()
    plt.savefig('data_analysis.png')
    print("\n분석 그래프가 'data_analysis.png'에 저장되었습니다.")
    
    # 상태 데이터 샘플 확인
    print("\n=== 상태 데이터 샘플 ===")
    print("상태 벡터의 첫 10개 값:", X[0][:10])
    print("최소값:", X.min())
    print("최대값:", X.max())
    print("평균값:", X.mean())

if __name__ == "__main__":
    if not all(os.path.exists(f'./data/{file}') for file in ['X_train.npy', 'y_train.npy', 'scores.npy']):
        print("데이터 파일을 찾을 수 없습니다. pretrain.py를 먼저 실행해주세요.")
    else:
        view_training_data() 