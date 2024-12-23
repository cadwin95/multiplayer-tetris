import json
import numpy as np
from typing import List, Dict, Any
import zipfile
import os
import subprocess
import sys
import shutil

def change_to_project_root():
    """스크립트가 있는 디렉토리에서 프로젝트 루트로 이동합니다."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # ai 폴더의 상위 디렉토리가 프로젝트 루트
    os.chdir(project_root)
    print(f"작업 디렉토리를 프로젝트 루트로 변경: {project_root}")

def export_convex_data(output_path: str = './data.zip') -> None:
    """Convex 데이터를 ZIP 파일로 내보냅니다."""
    print("Convex 데이터 내보내는 중...")
    try:
        # 현재 작업 디렉토리를 프로젝트 루트로 변경
        change_to_project_root()
        
        # 기존 ZIP 파일이 있다면 삭제
        if os.path.exists(output_path):
            os.remove(output_path)
            print(f"기존 {output_path} 파일을 삭제했습니다.")
        
        subprocess.run(['npx', 'convex', 'export', '--path', output_path], check=True)
        print(f"데이터가 {output_path}에 성공적으로 내보내졌습니다.")
    except subprocess.CalledProcessError as e:
        print(f"Convex 데이터 내보내기 실패: {e}")
        raise

def load_game_history(zip_path: str = './data.zip') -> List[Dict[str, Any]]:
    """ZIP 파일에서 게임 히스토리 데이터를 로드합니다."""
    history_data = []
    
    # 임시 디렉토리 생성
    temp_dir = './ai/temp_extract'
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # ZIP 파일의 내용을 임시 디렉토리에 추출
            zip_ref.extractall(temp_dir)
            
            # documents.jsonl 파일 읽기
            jsonl_path = os.path.join(temp_dir, 'gameHistory', 'documents.jsonl')
            if os.path.exists(jsonl_path):
                with open(jsonl_path, 'r') as f:
                    # JSONL 형식: 각 줄이 하나의 JSON 객체
                    for line in f:
                        try:
                            if line.strip():  # 빈 줄 무시
                                data = json.loads(line)
                                history_data.append(data)
                        except json.JSONDecodeError as e:
                            print(f"JSON 파싱 오류 (무시하고 계속): {e}")
                            continue
    finally:
        # 임시 디렉토리 유지 (디버깅 목적)
        pass
        # if os.path.exists(temp_dir):
        #     shutil.rmtree(temp_dir)
    
    return history_data

def board_string_to_matrix(board_string: str, width: int = 10, height: int = 20) -> np.ndarray:
    """보드 문자열을 2D numpy 배열로 변환합니다."""
    board = np.array([int(x) for x in board_string])
    return board.reshape(height, width)

# Tetris 피스와 액션을 위한 매핑 추가
piece_map = {
    'I': 0,
    'O': 1,
    'T': 2,
    'S': 3,
    'Z': 4,
    'J': 5,
    'L': 6
}

action_map = {
    'left': 0,
    'right': 1,
    'rotate': 2,
    'down': 3,
    'hardDrop': 4,
    'hold': 5
}

def create_training_data(history):
    X = []
    y = []
    
    for entry in history:
        try:
            # 1. 보드 상태 (200)
            board = board_string_to_matrix(entry['board']).flatten()
            
            # 2. 현재 피스 위치 매트릭스 생성 (200)
            current_piece_matrix = np.zeros((20, 10))
            piece_type = entry['pieceType']
            pos = entry['position']
            rotation = entry['rotation']
            # TODO: 피스 모양과 회전을 고려하여 매트릭스에 표시
            current_piece = current_piece_matrix.flatten()
            
            # 3. 착지 위치 매트릭스 생성 (200)
            landing_matrix = np.zeros((20, 10))
            # TODO: 현재 피스의 착지 위치 계산
            landing = landing_matrix.flatten()
            
            # 4. 다음 피스 원-핫 인코딩 (7)
            next_piece = np.zeros(7)
            next_piece[piece_map[entry['nextPiece']]] = 1
            
            # 특징 결합
            features = np.concatenate([
                board,           # 200
                current_piece,   # 200
                landing,        # 200
                next_piece      # 7
            ])
            
            # 레이블 수정 - dtype을 명시적으로 지정
            action = np.array(action_map[entry['action']], dtype=np.int64)
            
            X.append(features)
            y.append(action)
            
        except Exception as e:
            print(f"데이터 처리 중 오류 발생: {e}")
            continue
        print(X[0],y[0])
    return np.array(X), np.array(y, dtype=np.int64)  # y의 dtype을 명시적으로 지정

def main():
    try:
        # Convex 데이터 내보내기
        export_convex_data()
        
        # 게임 히스토리 데이터 가져오기
        print("게임 히스토리 데이터를 가져오는 중...")
        history = load_game_history()
        print(f"가져온 게임 히스토리 수: {len(history)}")
        
        # 학습 데이터 생성
        print("학습 데이터 생성 중...")
        X, y = create_training_data(history)
        
        # 데이터 저장
        print("데이터 저장 중...")
        os.makedirs('./ai/data', exist_ok=True)
        np.save('./ai/data/X_train.npy', X)
        np.save('./ai/data/y_train.npy', y)
        
        print(f"\n생성된 학습 데이터 크기:")
        print(f"입력 (X): {X.shape}")
        print(f"출력 (y): {y.shape}")
        
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == '__main__':
    main() 