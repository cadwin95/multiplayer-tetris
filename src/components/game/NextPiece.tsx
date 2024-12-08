// src/components/game/NextPiece.tsx
interface NextPieceProps {
  piece: string;  // 'I', 'O', 'T', 'S', 'Z', 'J', 'L' 중 하나
}

export function NextPiece({ piece }: NextPieceProps) {
  // 테트로미노 모양 정의
  const pieceMatrices: { [key: string]: number[][] } = {
    'I': [
      [1, 1, 1, 1]
    ],
    'O': [
      [1, 1],
      [1, 1]
    ],
    'T': [
      [0, 1, 0],
      [1, 1, 1]
    ],
    'S': [
      [0, 1, 1],
      [1, 1, 0]
    ],
    'Z': [
      [1, 1, 0],
      [0, 1, 1]
    ],
    'J': [
      [1, 0, 0],
      [1, 1, 1]
    ],
    'L': [
      [0, 0, 1],
      [1, 1, 1]
    ]
  };

  const pieceMatrix = pieceMatrices[piece] || [[0]];

  return (
    <div className="mt-4">
      <h3 className="text-white text-center mb-2">Next Piece</h3>
      <div className="bg-gray-900 p-2 rounded-lg">
        <div className="grid gap-px" 
             style={{ gridTemplateColumns: `repeat(${pieceMatrix[0].length}, 1fr)` }}>
          {pieceMatrix.map((row, i) => 
            row.map((cell, j) => (
              <div 
                key={`${i}-${j}`}
                className={`w-4 h-4 ${cell ? 'bg-blue-500' : 'bg-gray-800'}`}
              />
            ))
          )}
        </div>
      </div>
    </div>
  );
}