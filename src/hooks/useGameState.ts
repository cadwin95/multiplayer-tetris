import { useMutation, useQuery } from "convex/react";
import { api } from "../../convex/_generated/api";  
import { DirectionType, PieceType, GameStatus } from '../../convex/schema';
import { Id } from "../../convex/_generated/dataModel";
import { useState } from "react";

interface Piece {
  type: PieceType;
  rotation: number;
  position: { x: number; y: number };
}

interface GameState {
  board: string;
  currentPiece: Piece;
  nextPiece: Piece;
  holdPiece: Piece | null;
  score: number;
  level: number;
  lines: number;
  status: GameStatus;
}

interface GameStateHook {
  state: GameState;
  move: (direction: DirectionType) => Promise<void>;
  error: Error | null;
  isLoading: boolean;
}

// GameAction 결과 타입 정의 추가
interface GameActionResult {
  clearedLines: number;
  success: boolean;
}

export function useGameState(gameId: Id<"games">, playerId: Id<"players">): GameStateHook {
  const player = useQuery(api.games.getPlayer, { playerId });
  const game = useQuery(api.games.getGame, { gameId });
  const handleGameAction = useMutation(api.games.handleGameAction);
  const [error, setError] = useState<Error | null>(null);

  const state: GameState = {
    board: player?.board ?? "0".repeat(200),
    currentPiece: {
      type: (player?.currentPiece ?? 'I') as PieceType,
      rotation: player?.rotation ?? 0,
      position: player?.position ?? { x: 4, y: 0 }
    },
    nextPiece: {
      type: (player?.nextPiece ?? 'I') as PieceType,
      rotation: 0,
      position: { x: 4, y: 0 }
    },
    holdPiece: player?.holdPiece ? {
      type: player.holdPiece as PieceType,
      rotation: 0,
      position: { x: 4, y: 0 }
    } : null,
    score: Math.max(0, player?.score ?? 0),
    level: Math.max(1, player?.level ?? 1),
    lines: Math.max(0, player?.lines ?? 0),
    status: game?.status ?? 'waiting'
  };

  const move = async (direction: DirectionType) => {
    try {
      setError(null);
      const result = (await handleGameAction({
        gameId,
        playerId,
        action: direction
      }) as unknown) as GameActionResult;  // unknown을 통해 안전하게 타입 변환

      if (result?.clearedLines > 0) {
        // 서버에서 자동으로 점수가 업데이트되므로,
        // 여기서는 추가 작업이 필요 없음
      }
    } catch (error) {
      console.error('Move failed:', error);
      setError(error instanceof Error ? error : new Error('Unknown error'));
    }
  };

  return {
    state,
    move,
    error,
    isLoading: !player || !game
  };
}