// src/components/game/NextPiece.tsx
import { Piece, PIECE_ROTATIONS, PIECE_COLORS } from '../../../convex/schema';

interface NextPieceProps {
  piece: Piece;
}

export function NextPiece({ piece }: NextPieceProps) {
  const pieceMatrix = PIECE_ROTATIONS[piece.type][0];

  return (
    <div className="next-piece">
      <h3 className="text-white mb-2">Next</h3>
      <div className="piece-preview bg-gray-800 p-2 rounded">
        <div className="grid gap-1">
          {pieceMatrix.map((row: number[], i: number) => (
            <div key={i} className="flex gap-1">
              {row.map((cell: number, j: number) => (
                <div
                  key={`${i}-${j}`}
                  className={`w-4 h-4 rounded ${cell ? '' : 'bg-transparent'}`}
                  style={{
                    backgroundColor: cell ? PIECE_COLORS[piece.type] : undefined
                  }}
                />
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}