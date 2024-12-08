interface GameBoardProps {
  board: string;
  currentPiece: string;
  position: { x: number; y: number };
}

export function GameBoard({ board, currentPiece, position }: GameBoardProps) {
  console.log('GameBoard props:', { board, currentPiece, position });

  const boardMatrix = Array(20).fill(null).map((_, i) => 
    board.slice(i * 10, (i + 1) * 10).split('').map(Number)
  );

  const pieceMatrix = getPieceMatrix(currentPiece);
  if (position && pieceMatrix) {
    for (let y = 0; y < pieceMatrix.length; y++) {
      for (let x = 0; x < pieceMatrix[y].length; x++) {
        if (pieceMatrix[y][x]) {
          const boardY = position.y + y;
          const boardX = position.x + x;
          if (boardY >= 0 && boardY < 20 && boardX >= 0 && boardX < 10) {
            boardMatrix[boardY][boardX] = 2;
          }
        }
      }
    }
  }

  return (
    <div className="bg-gray-900 p-4 rounded-lg border border-gray-700">
      <div className="grid grid-cols-10 gap-[2px] bg-gray-800 p-2">
        {boardMatrix.map((row, i) => 
          row.map((cell, j) => (
            <div 
              key={`${i}-${j}`}
              className={`
                w-8 h-8 rounded-sm
                ${cell === 0 ? 'bg-gray-900' : 
                  cell === 1 ? 'bg-blue-500' :
                  'bg-purple-500'
                }
                ${cell > 0 ? 'border border-white/20' : ''}
              `}
            />
          ))
        )}
      </div>
    </div>
  );
}

function getPieceMatrix(piece: string): number[][] {
  const shapes: { [key: string]: number[][] } = {
    'I0': [[1, 1, 1, 1]],
    'I1': [[1], [1], [1], [1]],
    'O': [[1, 1], [1, 1]],
    'T0': [[0, 1, 0], [1, 1, 1]],
    'T1': [[1, 0], [1, 1], [1, 0]],
    'T2': [[1, 1, 1], [0, 1, 0]],
    'T3': [[0, 1], [1, 1], [0, 1]],
    'S0': [[0, 1, 1], [1, 1, 0]],
    'S1': [[1, 0], [1, 1], [0, 1]],
    'Z0': [[1, 1, 0], [0, 1, 1]],
    'Z1': [[0, 1], [1, 1], [1, 0]],
    'J0': [[1, 0, 0], [1, 1, 1]],
    'J1': [[1, 1], [1, 0], [1, 0]],
    'J2': [[1, 1, 1], [0, 0, 1]],
    'J3': [[0, 1], [0, 1], [1, 1]],
    'L0': [[0, 0, 1], [1, 1, 1]],
    'L1': [[1, 0], [1, 0], [1, 1]],
    'L2': [[1, 1, 1], [1, 0, 0]],
    'L3': [[1, 1], [0, 1], [0, 1]]
  };
  return shapes[piece] || [[1]];
}