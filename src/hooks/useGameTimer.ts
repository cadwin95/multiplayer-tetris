// hooks/useGameTimer.ts
import { useEffect } from 'react';
import { useMutation } from "convex/react";
import { api } from "../../convex/_generated/api";
import { Id } from "../../convex/_generated/dataModel";

export function useGameTimer(
  isPlaying: boolean,
  gameId: Id<"games">,
  level: number,
  isAIPlayer: boolean = false
) {
  const moveTetrominoe = useMutation(api.games.handleGameAction);
  const playerId = localStorage.getItem('playerId') as Id<"players">;

  useEffect(() => {
    if (!isPlaying || (isAIPlayer && !playerId)) return;

    if (!isAIPlayer) {
      const interval = setInterval(async () => {
        await moveTetrominoe({
          gameId,
          playerId,
          action: 'down'
        });
      }, Math.max(100, 1000 - (level * 100)));

      return () => clearInterval(interval);
    }
  }, [isPlaying, gameId, level, playerId, moveTetrominoe, isAIPlayer]);
}