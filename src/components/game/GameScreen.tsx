import { useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { Id } from "../../../convex/_generated/dataModel";
import { useGameState } from '../../hooks/useGameState';
import { useKeyboard } from '../../hooks/useKeyboard';
import { GameOver } from './GameOver';
import { useGameTimer } from '../../hooks/useGameTimer';
import { usePerformanceMonitor } from '../../hooks/usePerformanceMonitor';
import { MiniBoard } from './MiniBoard';
import { PIECE_ROTATIONS, PIECE_COLORS } from '../../../convex/schema';
import { useQuery } from "convex/react";
import { api } from "../../../convex/_generated/api";
import { Doc } from "../../../convex/_generated/dataModel";

interface GameScreenProps {
  gameId: Id<"games">;
  playerId: Id<"players">;
  isMinimized?: boolean;
}

export function GameScreen({ gameId, playerId, isMinimized = false }: GameScreenProps) {
  const { state, error, isLoading, actions } = useGameState(gameId, playerId);
  const navigate = useNavigate();
  const { startMeasurement, measureLatency } = usePerformanceMonitor();
  const game = useQuery(api.games.getGame, { gameId });
  const isAIPlayer = (game?.players as Doc<"players">[] | undefined)?.find(p => p._id === playerId)?.isAI;

  // Hold/Next 피스 props 준비
  const holdPieceProps = {
    board: "0".repeat(40),
    currentPiece: state.holdPiece ? {
      ...state.holdPiece,
      rotation: 0,
      position: { x: 2, y: 1 }
    } : undefined
  };

  const nextPieceProps = {
    board: "0".repeat(40),
    currentPiece: {
      ...state.nextPiece,
      rotation: 0,
      position: { x: 2, y: 1 }
    }
  };

  useGameTimer(
    state.status === 'playing',
    gameId,
    state.level,
    isAIPlayer
  );

  useKeyboard({
    isEnabled: !isAIPlayer && state.status === 'playing' && !isMinimized,
    onMoveLeft: () => {
      startMeasurement();
      actions.move('left');
    },
    onMoveRight: () => {
      startMeasurement();
      actions.move('right');
    },
    onMoveDown: () => {
      startMeasurement();
      actions.move('down');
    },
    onRotate: () => {
      startMeasurement();
      actions.move('rotate');
    },
    onHardDrop: () => {
      startMeasurement();
      actions.move('hardDrop');
    },
    onHold: () => {
      startMeasurement();
      actions.move('hold');
    }
  });

  useEffect(() => {
    if (state.status === 'finished') {
      const timer = setTimeout(() => {
        const playerId = localStorage.getItem('playerId');
        localStorage.clear();
        if (playerId) {
          localStorage.setItem('playerId', playerId);
        }
        navigate('/', { replace: true });
      }, 3000);

      return () => clearTimeout(timer);
    }
  }, [state.status, navigate]);

  useEffect(() => {
    if (state.status === 'playing') {
      measureLatency();
    }
  }, [state.board, state.status, measureLatency]);

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

  if (isLoading) return <div className="text-white text-center">Loading...</div>;
  if (error) return <div className="text-white text-center">Error: {error?.message}</div>;
  if (!state) return <div className="text-white text-center">No game state</div>;

  return (
    <div className={`game-screen ${isMinimized ? 'minimized' : ''}`}>
      <div className={isMinimized ? 'mini-game-layout' : 'game-layout'}>
        {/* 왼쪽 사이드 패널 */}
        <div className={isMinimized ? 'mini-side-panel' : 'side-panel'}>
          <div className={isMinimized ? 'mini-preview-box' : 'preview-box'}>
            <div className={isMinimized ? 'mini-preview-title' : 'preview-box-title'}>HOLD</div>
            {state.holdPiece && <MiniBoard {...holdPieceProps} />}
          </div>
        </div>

        {/* 메인 보드 */}
        <div className={isMinimized ? 'mini-board' : 'main-board'}>
          <div className="game-grid">
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
                    className={`game-cell ${isPieceCell || cell === '1' ? 'filled' : ''} ${isGhostCell ? 'ghost' : ''}`}
                    style={{
                      backgroundColor: isPieceCell ? PIECE_COLORS[state.currentPiece!.type as keyof typeof PIECE_COLORS] :
                                     cell === '1' ? '#4A5568' : 'transparent',
                      borderColor: isGhostCell ? PIECE_COLORS[ghostPiecePosition!.type as keyof typeof PIECE_COLORS] : undefined
                    }}
                  />
                );
              })
            )}
          </div>
          {!isMinimized && (
            <div className="score-panel">
              <div className="flex justify-between text-white">
                <div>SCORE: {state.score}</div>
                <div>LEVEL: {state.level}</div>
              </div>
            </div>
          )}
        </div>

        {/* 오른쪽 사이드 패널 */}
        <div className={isMinimized ? 'mini-side-panel' : 'side-panel'}>
          <div className={isMinimized ? 'mini-preview-box' : 'preview-box'}>
            <div className={isMinimized ? 'mini-preview-title' : 'preview-box-title'}>NEXT</div>
            <MiniBoard {...nextPieceProps} />
          </div>
          {!isMinimized && (
            <div className="preview-box">
              <div className="controls-guide">
                {/* 컨트롤 가이드 */}
              </div>
            </div>
          )}
        </div>
      </div>

      {state.status === 'finished' && (
        <GameOver 
          score={state.score}
          winnerId={game?.winnerId}
          playerId={playerId}
        />
      )}
    </div>
  );
}