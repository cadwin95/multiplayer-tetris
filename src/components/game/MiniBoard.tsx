import { Piece, PIECE_ROTATIONS, PIECE_COLORS } from '../../../convex/schema';

interface MiniBoardProps {
  board: string;
  currentPiece?: Piece;
}

export function MiniBoard({ board, currentPiece }: MiniBoardProps) {
  const boardMatrix = Array(20).fill(null).map((_, i) => 
    board.slice(i * 10, (i + 1) * 10).split('').map(Number)
  );

  // 현재 피스 그리기
  if (currentPiece) {
    const pieceMatrix = PIECE_ROTATIONS[currentPiece.type][currentPiece.rotation];
    for (let y = 0; y < pieceMatrix.length; y++) {
      for (let x = 0; x < pieceMatrix[y].length; x++) {
        if (pieceMatrix[y][x]) {
          const boardY = currentPiece.position.y + y;
          const boardX = currentPiece.position.x + x;
          if (boardY >= 0 && boardY < 20 && boardX >= 0 && boardX < 10) {
            boardMatrix[boardY][boardX] = 2;  // 2는 현재 피스를 나타냄
          }
        }
      }
    }
  }

  return (
    <div className="grid grid-cols-10 gap-px bg-gray-700 p-px">
      {boardMatrix.flat().map((cell, i) => (
        <div
          key={i}
          className={`aspect-square ${
            cell === 0 ? 'bg-gray-900' : 'bg-blue-500'
          }`}
          style={{
            backgroundColor: cell === 2 && currentPiece 
              ? PIECE_COLORS[currentPiece.type] 
              : cell === 1 
                ? '#4A5568' 
                : '#1A202C'
          }}
        />
      ))}
    </div>
  );
} 