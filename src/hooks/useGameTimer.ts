// hooks/useGameTimer.ts
import { useEffect, useRef } from 'react';

export function useGameTimer(
  onTick: () => void,
  level: number,
  isEnabled: boolean = true
) {
  const lastTickRef = useRef<number>(0);
  
  useEffect(() => {
    if (!isEnabled) return;

    const dropSpeed = Math.max(50, 800 - (level * 50));
    
    const tick = () => {
      const now = Date.now();
      if (now - lastTickRef.current >= dropSpeed) {
        onTick();
        lastTickRef.current = now;
      }
      requestAnimationFrame(tick);
    };

    lastTickRef.current = Date.now();
    const animationId = requestAnimationFrame(tick);
    
    return () => {
      cancelAnimationFrame(animationId);
    };
  }, [isEnabled, level, onTick]);
}