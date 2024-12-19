import { Piece, PIECE_ROTATIONS, PIECE_COLORS } from '../../../convex/schema';
import { MiniBoard } from './MiniBoard';

interface GameBoardProps {
  board: string;
  currentPiece: Piece;
  nextPiece: Piece;
  holdPiece?: Piece | null;
  score: number;
  level: number;
  lines: number;
}

export function GameBoard({ 
  board, 
  currentPiece, 
  nextPiece, 
  holdPiece,
  score,
  level,
  lines
}: GameBoardProps) {
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
            boardMatrix[boardY][boardX] = 2;
          }
        }
      }
    }
  }

  return (
    <div className="flex gap-6">
      {/* Side Info - Left */}
      <div className="w-32">
        <div className="bg-gray-800 p-4 rounded-lg mb-4">
          <h3 className="text-white text-lg mb-2">Hold</h3>
          <div className="aspect-square bg-gray-900 p-2 rounded">
            {holdPiece && (
              <MiniBoard 
                board={"0".repeat(40)}
                currentPiece={{
                  ...holdPiece,
                  rotation: 0,
                  position: { x: 2, y: 1 }
                }}
              />
            )}
          </div>
        </div>
      </div>

      {/* Main Board */}
      <div className="flex-1 max-w-md">
        <div className="aspect-[1/2] bg-gray-800 p-2 rounded-lg">
          <div className="h-full grid grid-cols-10 gap-px bg-gray-700">
            {boardMatrix.flat().map((cell, i) => (
              <div
                key={i}
                className="aspect-square"
                style={{
                  backgroundColor: cell === 2 ? PIECE_COLORS[currentPiece.type] :
                                 cell === 1 ? '#4A5568' : '#1A202C'
                }}
              />
            ))}
          </div>
        </div>
      </div>

      {/* Side Info - Right */}
      <div className="w-32">
        <div className="bg-gray-800 p-4 rounded-lg mb-4">
          <h3 className="text-white text-lg mb-2">Next</h3>
          <div className="aspect-square bg-gray-900 p-2 rounded">
            <MiniBoard 
              board={"0".repeat(40)}
              currentPiece={{
                ...nextPiece,
                rotation: 0,
                position: { x: 2, y: 1 }
              }}
            />
          </div>
        </div>
        <div className="bg-gray-800 p-4 rounded-lg">
          <div className="text-white mb-2">Score: {score}</div>
          <div className="text-white mb-2">Level: {level}</div>
          <div className="text-white">Lines: {lines}</div>
        </div>
      </div>
    </div>
  );
}