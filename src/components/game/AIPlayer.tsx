import { useEffect, useRef } from 'react';
import { Id } from "../../../convex/_generated/dataModel";
import { useGameState } from '../../hooks/useGameState';
import { useMutation } from "convex/react";
import { api } from "../../../convex/_generated/api";
import { ONNXTetrisAI } from '../../ai/ONNXTetrisAI';

interface AIPlayerProps {
  gameId: Id<"games">;
  playerId: Id<"players">;
}

export function AIPlayer({ gameId, playerId }: AIPlayerProps) {
  const { state } = useGameState(gameId, playerId);
  const moveTetrominoe = useMutation(api.games.handleGameAction);
  const aiRef = useRef<ONNXTetrisAI | null>(null);

  useEffect(() => {
    if (!aiRef.current) {
      aiRef.current = new ONNXTetrisAI();
    }
  }, []);

  useEffect(() => {
    if (state.status === 'playing' && aiRef.current) {
      const interval = setInterval(async () => {
        const action = await aiRef.current!.predictNextMove({
          board: state.board,
          currentPiece: state.currentPiece!,
          nextPiece: state.nextPiece,
          holdPiece: state.holdPiece,
          status: state.status,
          score: state.score,
          level: state.level,
          lines: state.lines
        });

        await moveTetrominoe({
          gameId,
          playerId,
          action
        });
      }, Math.max(50, 300 - (state.level * 20)));

      return () => clearInterval(interval);
    }
  }, [state, moveTetrominoe, gameId, playerId]);

  return null;
} 