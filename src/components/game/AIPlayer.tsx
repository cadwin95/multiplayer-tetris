import { useEffect, useRef, useState } from 'react';
import { Id } from "../../../convex/_generated/dataModel";
import { useGameState } from '../../hooks/useGameState';
import { useMutation } from "convex/react";
import { api } from "../../../convex/_generated/api";
import { HeuristicTetrisAI } from '../../ai/HeuristicTetrisAI';
import { GameScreen } from './GameScreen';

interface AIPlayerProps {
  gameId: Id<"games">;
  playerId: Id<"players">;
  isLoading?: boolean;
}

export function AIPlayer({ gameId, playerId, isLoading }: AIPlayerProps) {
  const { state } = useGameState(gameId, playerId);
  const moveTetrominoe = useMutation(api.games.handleGameAction);
  const aiRef = useRef<HeuristicTetrisAI | null>(null);
  const [lastMoveTime, setLastMoveTime] = useState<number>(Date.now());
  const [consecutiveMoves, setConsecutiveMoves] = useState<number>(0);

  // AI 초기화
  useEffect(() => {
    if (!aiRef.current) {
      aiRef.current = new HeuristicTetrisAI();
    }
  }, []);

  // AI 의사결정 로직
  useEffect(() => {
    if (state.status === 'playing' && aiRef.current) {
      const interval = setInterval(async () => {
        const now = Date.now();
        const timeSinceLastMove = now - lastMoveTime;

        // 연속 동작 제한 및 쿨다운 적용
        if (timeSinceLastMove < 100) {
          return;
        }

        if (consecutiveMoves > 10) {
          if (timeSinceLastMove < 500) {
            return;
          }
          setConsecutiveMoves(0);
        }

        const action = await aiRef.current!.predictNextMove({
          board: state.board,
          currentPiece: state.currentPiece,
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

        setLastMoveTime(now);
        setConsecutiveMoves(prev => prev + 1);
      }, Math.max(50, 300 - (state.level * 20))); // AI 속도 조절

      return () => clearInterval(interval);
    }
  }, [state, moveTetrominoe, gameId, playerId, lastMoveTime, consecutiveMoves]);

  return (
    <div className="relative">
      <GameScreen
        gameId={gameId}
        playerId={playerId}
        isMinimized={false}
        isLoading={isLoading}
      />
      <div className="absolute top-4 left-1/2 -translate-x-1/2 text-white text-center text-xl font-mono z-10 border border-white p-2 rounded-lg">
        AI Player
      </div>
    </div>
  );
} 