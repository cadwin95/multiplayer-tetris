// hooks/useGameTimer.ts
import { useEffect } from 'react';
import { Id } from "../../convex/_generated/dataModel";
import { useMutation } from "convex/react";
import { api } from "../../convex/_generated/api";

export function useGameTimer(
  isPlaying: boolean,
  gameId: Id<"games">,
  playerId: Id<"players">,
  level: number
) {
  const autoDown = useMutation(api.games.autoDown);
  
  useEffect(() => {
    if (!isPlaying) return;

    // 레벨에 따른 하강 속도 계산 (ms)
    const speed = Math.max(100, 1000 - (level - 1) * 100); // 레벨당 100ms씩 빨라짐
    
    const timer = setInterval(async () => {
      try {
        await autoDown({
          gameId,
          playerId,
        });
      } catch (error) {
        console.error('Auto down failed:', error);
      }
    }, speed);

    return () => clearInterval(timer);
  }, [isPlaying, level, gameId, playerId, autoDown]);
}