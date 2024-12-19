import { PieceType, PIECE_ROTATIONS, PIECE_COLORS } from '../../../convex/schema';

interface HoldPieceProps {
  piece: PieceType | null;
}

export function HoldPiece({ piece }: HoldPieceProps) {
  const getPieceMatrix = () => {
    if (!piece) return null;
    return PIECE_ROTATIONS[piece][0] || null;
  };

  const matrix = getPieceMatrix();

  return (
    <div className="hold-piece">
      <h3 className="text-white mb-2">Hold</h3>
      <div className="piece-preview bg-gray-800 p-2 rounded">
        {matrix ? (
          <div className="grid gap-1">
            {matrix.map((row: number[], i: number) => (
              <div key={i} className="flex gap-1">
                {row.map((cell: number, j: number) => (
                  <div
                    key={`${i}-${j}`}
                    className={`w-4 h-4 rounded ${cell ? '' : 'bg-transparent'}`}
                    style={{
                      backgroundColor: cell && piece ? PIECE_COLORS[piece] : undefined
                    }}
                  />
                ))}
              </div>
            ))}
          </div>
        ) : (
          <div className="text-gray-400 text-sm">Empty</div>
        )}
      </div>
    </div>
  );
} 