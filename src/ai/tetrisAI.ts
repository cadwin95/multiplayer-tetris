import { ClientGameState } from '../../convex/schema';

type GameState = ClientGameState;
type Move = { type: 'none' | 'left' | 'right' | 'rotate' | 'drop' };

export class TetrisAI {
  private calculateScore(gameState: GameState): number {
    const { board, currentPiece } = gameState;
    let score = 0;
    
    if (currentPiece) {
      // 현재 높이에 따른 페널티
      score -= currentPiece.position.y * 2;
      
      // 보드 상태를 고려한 점수 계산
      const boardArray = Array(20).fill(null).map((_, i) => 
        board.slice(i * 10, (i + 1) * 10).split('').map(Number)
      );
      
      // 높이 페널티
      const heights = boardArray[0].map((_, x) => {
        for (let y = 0; y < 20; y++) {
          if (boardArray[y][x] === 1) return 20 - y;
        }
        return 0;
      });
      score -= Math.max(...heights) * 2;
      
      // 구멍 페널티
      let holes = 0;
      for (let x = 0; x < 10; x++) {
        let foundBlock = false;
        for (let y = 0; y < 20; y++) {
          if (boardArray[y][x] === 1) foundBlock = true;
          else if (foundBlock) holes++;
        }
      }
      score -= holes * 10;
    }
    
    return score;
  }

  private getPossibleMoves(gameState: GameState): Move[] {
    const moves: Move[] = [];
    const { currentPiece } = gameState;
    
    if (currentPiece) {
      moves.push({ type: 'left' });
      moves.push({ type: 'right' });
      moves.push({ type: 'rotate' });
      moves.push({ type: 'drop' });
    }
    
    return moves;
  }

  private simulateMove(gameState: GameState, move: Move): GameState {
    const newState = { ...gameState };
    
    if (newState.currentPiece) {
      switch (move.type) {
        case 'left':
          newState.currentPiece.position.x--;
          break;
        case 'right':
          newState.currentPiece.position.x++;
          break;
        case 'rotate':
          newState.currentPiece.rotation = (newState.currentPiece.rotation + 1) % 4;
          break;
        case 'drop':
          newState.currentPiece.position.y++;
          break;
      }
    }
    
    return newState;
  }

  private async executeMove(move: Move): Promise<void> {
    // 실제 이동 실행
    console.log('Executing move:', move.type);
  }

  private evaluateMove(gameState: GameState): number {
    return this.calculateScore(gameState);
  }

  private findBestMove(gameState: GameState): Move {
    let bestScore = -Infinity;
    let bestMove: Move = { type: 'none' };

    for (const move of this.getPossibleMoves(gameState)) {
      const newState = this.simulateMove(gameState, move);
      const score = this.evaluateMove(newState);
      
      if (score > bestScore) {
        bestScore = score;
        bestMove = move;
      }
    }

    return bestMove;
  }

  public async playNextMove(gameState: GameState): Promise<void> {
    const bestMove = this.findBestMove(gameState);
    await this.executeMove(bestMove);
  }
} 