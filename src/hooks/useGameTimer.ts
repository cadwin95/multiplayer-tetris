// hooks/useGameTimer.ts
import { useEffect } from 'react';

export function useGameTimer(
  onTick: () => void,
  level: number,
  isEnabled: boolean = true
) {
  useEffect(() => {
    if (!isEnabled) return;

    const dropSpeed = Math.max(50, 800 - (level * 50));

    const interval = setInterval(onTick, dropSpeed);
    return () => clearInterval(interval);
  }, [isEnabled, level, onTick]);
}