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
    score: player?.score ?? 0,
    level: player?.level ?? 1,
    lines: player?.lines ?? 0,
    status: game?.status ?? 'waiting'
  };

  const move = async (direction: DirectionType) => {
    try {
      setError(null);
      await handleGameAction({
        gameId,
        playerId,
        action: direction
      });
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