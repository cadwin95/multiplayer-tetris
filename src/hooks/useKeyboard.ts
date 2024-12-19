// src/hooks/useKeyboard.ts
import { useEffect } from 'react';

interface KeyboardHandlers {
  isEnabled: boolean;
  onMoveLeft: () => void;
  onMoveRight: () => void;
  onMoveDown: () => void;
  onRotate: () => void;
  onHardDrop: () => void;
  onHold: () => void;
}

export function useKeyboard({
  isEnabled,
  onMoveLeft,
  onMoveRight,
  onMoveDown,
  onRotate,
  onHardDrop,
  onHold,
}: KeyboardHandlers) {
  useEffect(() => {
    if (!isEnabled) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.repeat) return;

      switch (event.code) {
        case 'ArrowLeft':
          event.preventDefault();
          onMoveLeft();
          break;
        case 'ArrowRight':
          event.preventDefault();
          onMoveRight();
          break;
        case 'ArrowDown':
          event.preventDefault();
          onMoveDown();
          break;
        case 'ArrowUp':
          event.preventDefault();
          onRotate();
          break;
        case 'Space':
          event.preventDefault();
          onHardDrop();
          break;
        case 'KeyC':
          event.preventDefault();
          onHold();
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isEnabled, onMoveLeft, onMoveRight, onMoveDown, onRotate, onHardDrop, onHold]);
}

