// src/hooks/useKeyboard.ts
import { useEffect, useCallback } from 'react';

interface KeyboardHandlers {
  onMoveLeft?: () => void;
  onMoveRight?: () => void;
  onMoveDown?: () => void;
  onRotate?: () => void;
  onHardDrop?: () => void;
  onPause?: () => void;
}

export function useKeyboard(
  isPlaying: boolean,
  handlers: KeyboardHandlers
) {
  const handleKeyDown = useCallback((event: KeyboardEvent) => {
    if (!isPlaying) return;

    switch (event.key) {
      case 'ArrowLeft':
        event.preventDefault();
        handlers.onMoveLeft?.();
        break;
      case 'ArrowRight':
        event.preventDefault();
        handlers.onMoveRight?.();
        break;
      case 'ArrowDown':
        event.preventDefault();
        handlers.onMoveDown?.();
        break;
      case 'ArrowUp':
        event.preventDefault();
        handlers.onRotate?.();
        break;
      case ' ': // Space bar
        event.preventDefault();
        handlers.onHardDrop?.();
        break;
      case 'p':
      case 'P':
        event.preventDefault();
        handlers.onPause?.();
        break;
    }
  }, [isPlaying, handlers]);

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [handleKeyDown]);
}

