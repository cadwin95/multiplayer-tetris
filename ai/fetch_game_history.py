import json
import numpy as np
from typing import List, Dict, Any
import zipfile
import os
import subprocess
import sys
import shutil
from environment import TetrisEnv

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
    print("\n=== 원본 데이터 검증 ===")
    action_counts = {}
    for entry in history:
        action = entry['action']
        action_counts[action] = action_counts.get(action, 0) + 1
    
    print("원본 액션 분포:")
    for action, count in action_counts.items():
        print(f"  - {action}: {count}개 ({count/len(history)*100:.1f}%)")
    
    env = TetrisEnv()
    X = []
    Y = []
    
    # 변환된 액션 카운트 추가
    converted_action_counts = {i: 0 for i in range(6)}
    
    for entry in history:
        try:
            if entry['action'] not in action_map:
                print(f"경고: 알 수 없는 액션 발견: {entry['action']}")
                continue
            
            # 1. 보드 상태 (200)
            board = board_string_to_matrix(entry['board']).flatten()
            
            # 2. 현재 피스 위치 매트릭스 생성 (200)
            current_piece_matrix = np.zeros((20, 10))
            piece_type = entry['pieceType']
            pos = [int(entry['position']['x']), int(entry['position']['y'])]
            rotation = int(entry['rotation'])
            
            # environment.py의 PIECES 사용
            piece_info = {
                'type': piece_type,
                'rotation': rotation,
                'position': pos,
                'shape': env.PIECES[piece_type][rotation % len(env.PIECES[piece_type])]
            }
            
            # 현재 피스 위치 표시
            shape = piece_info['shape']
            for y in range(len(shape)):
                for x in range(len(shape[0])):
                    if shape[y][x]:
                        board_y = pos[1] + y
                        board_x = pos[0] + x
                        if 0 <= board_y < 20 and 0 <= board_x < 10:
                            current_piece_matrix[board_y][board_x] = 1
            
            current_piece = current_piece_matrix.flatten()
            
            # 3. 착지 위치 매트릭스 생성 (200)
            landing_matrix = current_piece_matrix.copy()
            board_matrix = board_string_to_matrix(entry['board'])
            
            # 피스를 아래로 이동하면서 착지 위치 계산
            while not env._check_collision(
                {**piece_info, 'position': [
                    piece_info['position'][0],
                    piece_info['position'][1] + 1
                ]},
                board_matrix
            ):
                piece_info['position'][1] += 1
                landing_matrix = np.zeros((20, 10))
                for y in range(len(shape)):
                    for x in range(len(shape[0])):
                        if shape[y][x]:
                            board_y = piece_info['position'][1] + y
                            board_x = piece_info['position'][0] + x
                            if 0 <= board_y < 20 and 0 <= board_x < 10:
                                landing_matrix[board_y][board_x] = 1
            
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
            
            # 액션 매핑 디버깅
            original_action = entry['action']
            mapped_action = action_map[original_action]
            converted_action_counts[mapped_action] += 1
            
            if mapped_action not in range(6):
                print(f"경고: 잘못된 액션 값: {mapped_action} (원본: {original_action})")
                continue
            
            X.append(features)
            Y.append(mapped_action)
            
        except Exception as e:
            print(f"데이터 처리 중 오류 발생: {e}")
            continue
    
    print("\n변환된 액션 분포:")
    action_names = ['left', 'right', 'rotate', 'down', 'hardDrop', 'hold']
    for action_id, count in converted_action_counts.items():
        if len(Y) > 0:  # 분모가 0이 되는 것을 방지
            percentage = (count / len(Y)) * 100
            print(f"  - {action_names[action_id]}: {count}개 ({percentage:.1f}%)")
    
    return np.array(X), np.array(Y, dtype=np.int64)

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
        
        # 저장하기 전 데이터 검증
        print("\n=== 저장 전 최종 데이터 검증 ===")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print("\n액션 분포:")
        unique, counts = np.unique(y, return_counts=True)
        action_names = ['left', 'right', 'rotate', 'down', 'hardDrop', 'hold']
        for action_id, count in zip(unique, counts):
            percentage = (count / len(y)) * 100
            print(f"  - {action_names[action_id]}: {count}개 ({percentage:.1f}%)")
        
        # 데이터 저장
        print("\n데이터 저장 중...")
        save_dir = './ai/data'
        os.makedirs(save_dir, exist_ok=True)
        
        # 기존 파일 백업
        for filename in ['X_train.npy', 'y_train.npy']:
            filepath = os.path.join(save_dir, filename)
            if os.path.exists(filepath):
                backup_path = filepath + '.backup'
                os.rename(filepath, backup_path)
                print(f"기존 {filename} 파일을 {filename}.backup으로 백업했습니다.")
        
        # 새 데이터 저장
        np.save(os.path.join(save_dir, 'X_train.npy'), X)
        np.save(os.path.join(save_dir, 'y_train.npy'), y)
        
        print(f"\n생성된 학습 데이터 크기:")
        print(f"입력 (X): {X.shape}")
        print(f"출력 (y): {y.shape}")
        
        # 저장된 데이터 확인
        print("\n저장된 데이터 확인")
        
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == '__main__':
    main() 