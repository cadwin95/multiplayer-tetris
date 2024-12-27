import { useState, useEffect } from 'react';
import { MiniBoard, MiniBoardProps } from './MiniBoard';
import { PIECE_ROTATIONS, PieceType } from '../../../convex/schema';

interface FloatingBoard {
  id: number;
  x: number;
  y: number;
  speed: number;
  direction: number;
  piece: PieceType;
  rotation: number;
}

export function FloatingBoards() {
  const [boards, setBoards] = useState<FloatingBoard[]>([]);
  
  useEffect(() => {
    const pieces = Object.keys(PIECE_ROTATIONS) as PieceType[];
    const newBoards: FloatingBoard[] = Array(12).fill(null).map((_, i) => ({
      id: i,
      x: Math.random() * 100,
      y: Math.random() * 100,
      speed: 0.2 + Math.random() * 0.3,
      direction: Math.random() * Math.PI * 2,
      piece: pieces[Math.floor(Math.random() * pieces.length)],
      rotation: Math.floor(Math.random() * 4)
    }));
    setBoards(newBoards);
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      setBoards(prevBoards => prevBoards.map(board => ({
        ...board,
        x: (board.x + Math.cos(board.direction) * board.speed) % 100,
        y: (board.y + Math.sin(board.direction) * board.speed) % 100,
        direction: board.direction + (Math.random() - 0.5) * 0.1
      })));
    }, 50);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none">
      {boards.map(board => {
        const miniBoardProps: MiniBoardProps = {
          board: "0".repeat(16),
          currentPiece: {
            type: board.piece,
            rotation: board.rotation,
            position: { x: 0, y: 0 }
          },
          boardSize: { width: 4, height: 4 }
        };
        
        return (
          <div
            key={board.id}
            className="absolute w-16 h-16 transition-all duration-500 ease-linear opacity-40"
            style={{
              left: `${board.x}%`,
              top: `${board.y}%`,
              transform: `translate(-50%, -50%) rotate(${board.direction * 180 / Math.PI}deg)`,
              filter: 'drop-shadow(0 0 12px rgba(255, 255, 255, 0.4))',
            }}
          >
            <div className="bg-black/50 backdrop-blur-sm p-2 rounded-lg border border-white/20">
              <MiniBoard {...miniBoardProps} />
            </div>
          </div>
        );
      })}
    </div>
  );
} 