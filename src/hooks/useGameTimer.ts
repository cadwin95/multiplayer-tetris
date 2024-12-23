// hooks/useGameTimer.ts
import { useEffect } from 'react';

export function useGameTimer(
  onTick: () => void,
  level: number,
  isEnabled: boolean = true
) {
  useEffect(() => {
    if (!isEnabled) return;

    const interval = setInterval(
      onTick,
      Math.max(100, 1000 - (level * 100))
    );

    return () => clearInterval(interval);
  }, [isEnabled, level, onTick]);
}