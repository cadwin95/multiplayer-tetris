import { Piece, PIECE_ROTATIONS, PIECE_COLORS } from '../../../convex/schema';

export interface MiniBoardProps {
  board: string;
  currentPiece?: Piece;
  boardSize: {
    width: number;
    height: number;
  };
}

export function MiniBoard({ board, currentPiece, boardSize }: MiniBoardProps) {
  const boardMatrix = Array(boardSize.height).fill(null).map((_, i) => 
    board.slice(i * boardSize.width, (i + 1) * boardSize.width).split('').map(Number)
  );

  // 현재 피스 그리기
  if (currentPiece) {
    const pieceMatrix = PIECE_ROTATIONS[currentPiece.type][currentPiece.rotation];
    
    // 피스를 중앙에 배치하기 위한 오프셋 계산
    const offsetX = Math.floor((boardSize.width - pieceMatrix[0].length) / 2);
    const offsetY = Math.floor((boardSize.height - pieceMatrix.length) / 2);

    for (let y = 0; y < pieceMatrix.length; y++) {
      for (let x = 0; x < pieceMatrix[y].length; x++) {
        if (pieceMatrix[y][x]) {
          const boardY = offsetY + y;
          const boardX = offsetX + x;
          if (boardY >= 0 && boardY < boardSize.height && boardX >= 0 && boardX < boardSize.width) {
            boardMatrix[boardY][boardX] = 2;
          }
        }
      }
    }
  }

  return (
    <div 
      className="grid gap-[1px] bg-black/30 p-[1px] rounded-sm"
      style={{
        gridTemplateColumns: `repeat(${boardSize.width}, minmax(0, 1fr))`
      }}
    >
      {boardMatrix.flat().map((cell, i) => (
        <div
          key={i}
          className="aspect-square rounded-[1px]"
          style={{
            backgroundColor: cell === 2 && currentPiece 
              ? `${PIECE_COLORS[currentPiece.type]}66`
              : cell === 1 
                ? '#4A556855' 
                : '#1A202C22',
            boxShadow: cell === 2 && currentPiece
              ? `0 0 12px ${PIECE_COLORS[currentPiece.type]}99`
              : 'none'
          }}
        />
      ))}
    </div>
  );
} 