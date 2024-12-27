import { ClientGameState, DirectionType, Piece, PIECE_ROTATIONS } from '../../convex/schema';

interface EvaluationWeights {
  heightWeight: number;
  linesWeight: number;
  holesWeight: number;
  bumpinessWeight: number;
}

interface TargetPosition {
  x: number;
  rotation: number;
  score: number;
}

export class HeuristicTetrisAI {
  private weights: EvaluationWeights = {
    heightWeight: -0.510066,
    linesWeight: 0.760666,
    holesWeight: -0.35663,
    bumpinessWeight: -0.184483
  };

  private targetPosition: TargetPosition | null = null;

  private evaluatePosition(board: string, piece: Piece): number {
    const width = 10;
    const height = 20;
    const heights = new Array(width).fill(0);
    const holes = new Array(width).fill(0);
    let completedLines = 0;
    
    // 각 열의 높이와 구멍 계산
    for (let x = 0; x < width; x++) {
      let foundBlock = false;
      for (let y = 0; y < height; y++) {
        const cell = board[y * width + x];
        if (cell === '1') {
          foundBlock = true;
          heights[x] = Math.max(heights[x], height - y);
        } else if (foundBlock) {
          holes[x]++;
        }
      }
    }

    // 완성된 라인 계산
    for (let y = 0; y < height; y++) {
      let complete = true;
      for (let x = 0; x < width; x++) {
        if (board[y * width + x] === '0') {
          complete = false;
          break;
        }
      }
      if (complete) completedLines++;
    }

    // 높이 차이 계산
    let bumpiness = 0;
    for (let x = 0; x < width - 1; x++) {
      bumpiness += Math.abs(heights[x] - heights[x + 1]);
    }

    const totalHeight = heights.reduce((sum, h) => sum + h, 0);
    const totalHoles = holes.reduce((sum, h) => sum + h, 0);

    return this.weights.heightWeight * totalHeight +
           this.weights.linesWeight * completedLines +
           this.weights.holesWeight * totalHoles +
           this.weights.bumpinessWeight * bumpiness;
  }

  private findBestPosition(gameState: Partial<ClientGameState>): TargetPosition | null {
    if (!gameState.board || !gameState.currentPiece) return null;

    let bestScore = -Infinity;
    let bestPosition: TargetPosition | null = null;

    // 모든 회전과 x 위치에 대해 평가
    for (let rotation = 0; rotation < 4; rotation++) {
      const pieceMatrix = PIECE_ROTATIONS[gameState.currentPiece.type][rotation];
      const pieceWidth = pieceMatrix[0].length;

      // 가능한 모든 x 위치에 대해
      for (let x = 0; x < 10; x++) {
        const testPiece = {
          ...gameState.currentPiece,
          position: { x, y: 0 },
          rotation
        };

        // 이 위치의 점수 계산
        const score = this.evaluatePosition(gameState.board, testPiece);
        if (score > bestScore) {
          bestScore = score;
          bestPosition = { x, rotation, score };
        }
      }
    }

    return bestPosition;
  }

  public async predictNextMove(gameState: Partial<ClientGameState>): Promise<DirectionType> {
    if (!gameState.board || !gameState.currentPiece) {
      return 'down';
    }

    // 목표 위치가 없거나 새로운 피스라면 최적의 위치 계산
    if (!this.targetPosition || 
        gameState.currentPiece.position.y === 0) {
      this.targetPosition = this.findBestPosition(gameState);
    }

    if (!this.targetPosition) {
      return 'down';
    }

    // 현재 위치에서 목표 위치로 가기 위한 다음 행동 결정
    const currentPiece = gameState.currentPiece;

    // 회전이 필요한 경우
    if (currentPiece.rotation !== this.targetPosition.rotation) {
      return 'rotate';
    }

    // 수평 이동이 필요한 경우
    if (currentPiece.position.x < this.targetPosition.x) {
      return 'right';
    }
    if (currentPiece.position.x > this.targetPosition.x) {
      return 'left';
    }

    // 목표 x위치와 회전에 도달했다면 드롭
    return 'hardDrop';
  }
} 