import { useMutation, useQuery } from "convex/react";
import { api } from "../../convex/_generated/api";  
import { DirectionType, PieceType } from '../../convex/schema';
import { Id } from "../../convex/_generated/dataModel";

// Piece 타입 정의
interface Piece {
  type: PieceType;
  rotation: number;
  position: { x: number; y: number };
}

interface GameState {
  board: string;
  currentPiece: Piece | null;
  nextPiece: Piece;
  holdPiece: Piece | null;
  score: number;
  level: number;
  lines: number;
  status: 'playing' | 'paused' | 'finished' | 'waiting' | 'ready' | 'idle';
}

export function useGameState(gameId: Id<"games">, playerId: Id<"players">) {
  const game = useQuery(api.games.getGame, { gameId });
  const player = useQuery(api.games.getPlayer, { playerId });

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
    score: player?.score ?? 0,
    level: player?.level ?? 1,
    lines: player?.lines ?? 0,
    status: game?.status ?? 'waiting'
  };

  const moveTetrominoe = useMutation(api.games.handleGameAction);

  const actions = {
    move: async (direction: DirectionType) => {
      try {
        await moveTetrominoe({
          gameId,
          playerId,
          action: direction
        });
      } catch (error) {
        console.error('Move failed:', error);
      }
    }
  };

  return {
    state,
    error: null as Error | null,
    isLoading: !game || !player,
    actions
  };
}