import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Id } from "../../../convex/_generated/dataModel";
import { useGameState } from '../../hooks/useGameState';
import { useKeyboard } from '../../hooks/useKeyboard';
import { GameOver } from './GameOver';
import { useGameTimer } from '../../hooks/useGameTimer';
import { MiniBoard, MiniBoardProps } from './MiniBoard';
import { PIECE_ROTATIONS, PIECE_COLORS, PieceType } from '../../../convex/schema';
import { useQuery } from "convex/react";
import { api } from "../../../convex/_generated/api";
import { Doc } from "../../../convex/_generated/dataModel";

interface GameScreenProps {
  gameId: Id<"games">;
  playerId: Id<"players">;
  isMinimized?: boolean;
}

interface FloatingBoard {
  id: number;
  x: number;
  y: number;
  speed: number;
  direction: number;
  piece: PieceType;
  rotation: number;
}

export function GameScreen({ gameId, playerId, isMinimized = false }: GameScreenProps) {
  const { state, move, error, isLoading } = useGameState(gameId, playerId);
  const navigate = useNavigate();
  const game = useQuery(api.games.getGame, { gameId });
  const isAIPlayer = (game?.players as Doc<"players">[] | undefined)?.find(p => p._id === playerId)?.isAI;
  const [floatingBoards, setFloatingBoards] = useState<FloatingBoard[]>([]);

  // Hold/Next 피스 props 준비
  const holdPieceProps = {
    board: "0".repeat(16),
    currentPiece: state.holdPiece ? {
      ...state.holdPiece,
      rotation: 0,
      position: { x: 0, y: 0 }
    } : undefined,
    boardSize: { width: 4, height: 4 }
  };

  const nextPieceProps = {
    board: "0".repeat(16),
    currentPiece: {
      ...state.nextPiece,
      rotation: 0,
      position: { x: 0, y: 0 }
    },
    boardSize: { width: 4, height: 4 }
  };

  useGameTimer(
    () => {
      if (state.status === 'playing') {
        move('down');
      }
    },
    state.level,
    state.status === 'playing' && !isMinimized
  );

  useKeyboard({
    isEnabled: !isAIPlayer && state.status === 'playing' && !isMinimized,
    onMoveLeft: () => {
      move('left');
    },
    onMoveRight: () => {
      move('right');
    },
    onMoveDown: () => {
      move('down');
    },
    onRotate: () => {
      move('rotate');
    },
    onHardDrop: () => {
      move('hardDrop');
    },
    onHold: () => {
      move('hold');
    }
  });

  // 고스트 피스 위치 계산
  const ghostPiecePosition = useMemo(() => {
    if (!state.currentPiece) return null;
    
    let ghostY = state.currentPiece.position.y;
    const pieceMatrix = PIECE_ROTATIONS[state.currentPiece.type as keyof typeof PIECE_ROTATIONS][state.currentPiece.rotation];
    
    // 바닥까지 이동
    while (ghostY < 20) {
      let canMove = true;
      
      for (let y = 0; y < pieceMatrix.length; y++) {
        for (let x = 0; x < pieceMatrix[y].length; x++) {
          if (pieceMatrix[y][x]) {
            const boardY = ghostY + y;
            const boardX = state.currentPiece.position.x + x;
            
            if (boardY >= 20 || 
                boardX < 0 || 
                boardX >= 10 || 
                state.board[boardY * 10 + boardX] === '1') {
              canMove = false;
              break;
            }
          }
        }
        if (!canMove) break;
      }
      
      if (!canMove) break;
      ghostY++;
    }
    
    return {
      ...state.currentPiece,
      position: { ...state.currentPiece.position, y: ghostY - 1 }
    };
  }, [state.currentPiece, state.board]);

  // 상태 로깅 추가
  useEffect(() => {
    console.log('Game Status:', state.status);
    console.log('Game State:', state);
    console.log('Game Data:', game);
  }, [state.status, state, game]);

  // 떠다니는 미니보드 초기화
  useEffect(() => {
    const pieces = Object.keys(PIECE_ROTATIONS) as PieceType[];
    const boards: FloatingBoard[] = Array(8).fill(null).map((_, i) => ({
      id: i,
      x: Math.random() * 100,
      y: Math.random() * 100,
      speed: 0.2 + Math.random() * 0.3,
      direction: Math.random() * Math.PI * 2,
      piece: pieces[Math.floor(Math.random() * pieces.length)],
      rotation: Math.floor(Math.random() * 4)
    }));
    setFloatingBoards(boards);
  }, []);

  // 미니보드 움직임 애니메이션
  useEffect(() => {
    if (state.status !== 'playing') return;

    const interval = setInterval(() => {
      setFloatingBoards(prev => prev.map(board => {
        let newX = board.x + Math.cos(board.direction) * board.speed;
        let newY = board.y + Math.sin(board.direction) * board.speed;

        if (newX < 0 || newX > 100) {
          board.direction = Math.PI - board.direction;
          newX = board.x;
        }
        if (newY < 0 || newY > 100) {
          board.direction = -board.direction;
          newY = board.y;
        }

        return {
          ...board,
          x: newX,
          y: newY
        };
      }));
    }, 1000);

    return () => clearInterval(interval);
  }, [state.status]);

  if (isLoading) return <div className="text-white text-center">Loading...</div>;
  if (error) return <div className="text-white text-center">Error: {error?.message}</div>;
  if (!state) return <div className="text-white text-center">No game state</div>;

  return (
    <div className="game-screen h-screen w-screen flex items-center justify-center bg-[#1a1a2e]/90">
      {/* 배경 미니보드들 - 멀티플레이어 모드에서만 표시 */}
      {game?.mode === 'multi' && (
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          {floatingBoards.map(board => {
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
                className="absolute w-20 h-20 transition-all duration-[3000ms] ease-in-out opacity-30"
                style={{
                  left: `${board.x}%`,
                  top: `${board.y}%`,
                  transform: 'translate(-50%, -50%) rotate(' + (board.direction * 180 / Math.PI) + 'deg)',
                  filter: 'drop-shadow(0 0 8px rgba(255, 255, 255, 0.2))',
                }}
              >
                <MiniBoard {...miniBoardProps} />
              </div>
            );
          })}
        </div>
      )}

      <div className="relative h-[90vh] aspect-[3/4] bg-black/60 backdrop-blur-sm rounded-lg border border-white/10 shadow-2xl">
        {/* 사이드 정보 패널 */}
        <div className="absolute right-4 h-full py-4 flex flex-col justify-between">
          {/* Hold */}
          <div className="bg-black/50 backdrop-blur-md p-4 rounded-lg border border-white/10">
            <div className="text-gray-400 text-sm mb-2 font-mono">HOLD</div>
            <div className="w-24 h-24 border border-white/10 rounded-md overflow-hidden">
              {state.holdPiece && <MiniBoard {...holdPieceProps} />}
            </div>
          </div>

          {/* Next */}
          <div className="bg-black/50 backdrop-blur-md p-4 rounded-lg border border-white/10">
            <div className="text-gray-400 text-sm mb-2 font-mono">NEXT</div>
            <div className="w-24 h-24 border border-white/10 rounded-md overflow-hidden">
             {state.nextPiece && <MiniBoard {...nextPieceProps} />}
            </div>
          </div>

          {/* Score */}
          <div className="bg-black/50 backdrop-blur-md p-4 rounded-lg border border-white/10">
            <div className="text-gray-400 text-sm mb-2 font-mono">SCORE</div>
            <div className="text-2xl text-white font-mono">
              {state.score.toLocaleString()}
            </div>
            <div className="text-sm text-gray-400 mt-2">
              <div className="flex justify-between">
                <span>LEVEL</span>
                <span>{state.level}</span>
              </div>
              <div className="flex justify-between">
                <span>LINES</span>
                <span>{state.lines}</span>
              </div>
            </div>
          </div>
        </div>

        {/* 게임 제목 */}
        <div className="absolute top-4 left-1/2 -translate-x-1/2 text-xl text-gray-400 font-mono tracking-widest">
          TETRIS AI
        </div>

        {/* 메인 게임 보드 */}
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="game-grid h-full aspect-[1/2] border-[1px] border-white/10 rounded-lg overflow-hidden bg-black/50 backdrop-blur-md">
            {Array(20).fill(null).map((_, y) => 
              Array(10).fill(null).map((_, x) => {
                const cell = state.board[y * 10 + x];
                const isPieceCell = state.currentPiece &&
                  y >= state.currentPiece.position.y &&
                  y < state.currentPiece.position.y + PIECE_ROTATIONS[state.currentPiece.type as keyof typeof PIECE_ROTATIONS][state.currentPiece.rotation].length &&
                  x >= state.currentPiece.position.x &&
                  x < state.currentPiece.position.x + PIECE_ROTATIONS[state.currentPiece.type as keyof typeof PIECE_ROTATIONS][state.currentPiece.rotation][0].length &&
                  PIECE_ROTATIONS[state.currentPiece.type as keyof typeof PIECE_ROTATIONS][state.currentPiece.rotation][y - state.currentPiece.position.y][x - state.currentPiece.position.x];

                // 고스트 피스 셀 확인
                const isGhostCell = ghostPiecePosition &&
                  y >= ghostPiecePosition.position.y &&
                  y < ghostPiecePosition.position.y + PIECE_ROTATIONS[ghostPiecePosition.type as keyof typeof PIECE_ROTATIONS][ghostPiecePosition.rotation].length &&
                  x >= ghostPiecePosition.position.x &&
                  x < ghostPiecePosition.position.x + PIECE_ROTATIONS[ghostPiecePosition.type as keyof typeof PIECE_ROTATIONS][ghostPiecePosition.rotation][0].length &&
                  PIECE_ROTATIONS[ghostPiecePosition.type as keyof typeof PIECE_ROTATIONS][ghostPiecePosition.rotation][y - ghostPiecePosition.position.y][x - ghostPiecePosition.position.x];

                return (
                  <div
                    key={`${y}-${x}`}
                    className={`game-cell ${isPieceCell ? 'filled' : ''} ${
                      !isPieceCell && isGhostCell ? 'ghost' : ''
                    }`}
                    style={{
                      backgroundColor: isPieceCell ? PIECE_COLORS[state.currentPiece!.type as keyof typeof PIECE_COLORS] :
                             cell === '1' ? '#4A5568' : 'transparent'
                    }}
                  />
                );
              })
            )}
          </div>
        </div>
      </div>

      {/* GameOver 컴포넌트 */}
      {state.status === 'finished' && (
        <div className="absolute inset-0 bg-black/90 backdrop-blur-md flex items-center justify-center">
          <GameOver 
            score={state.score || 0}
            winnerId={game?.winnerId}
            playerId={playerId}
            onPlayAgain={() => navigate('/')}
          />
        </div>
      )}
    </div>
  );
}