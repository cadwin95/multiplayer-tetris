import { useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { Id } from "../../../convex/_generated/dataModel";
import { useGameState } from '../../hooks/useGameState';
import { GameOver } from './GameOver';
import { MiniBoard } from './MiniBoard';
import { PIECE_ROTATIONS, PIECE_COLORS } from '../../../convex/schema';
import { useQuery } from "convex/react";
import { api } from "../../../convex/_generated/api";
import { useGameTimer } from '../../hooks/useGameTimer';
import { useMutation } from "convex/react";




interface GameScreenProps {
  gameId: Id<"games">;
  playerId: Id<"players">;
  isMinimized?: boolean;
  isLoading?: boolean;
}



export function GameScreen({ gameId, playerId, isMinimized = false, isLoading = false }: GameScreenProps) {
  const { state, error, isLoading: gameStateLoading } = useGameState(gameId, playerId);
  const navigate = useNavigate();
  const game = useQuery(api.games.getGame, { gameId });
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
  
  const moveTetrominoe = useMutation(api.games.handleGameAction);   
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
    // 자동 하강 타이머
    useGameTimer(
      () => {
        if (state.status === 'playing') {
          moveTetrominoe({
            gameId,
            playerId,
            action: 'down'
          });
        }
      },
      state.level,
      state.status === 'playing' && !gameStateLoading
    );
  
  // 상태 로깅 추가
  useEffect(() => {
    console.log('Game Status:', state.status);
    console.log('Game State:', state);
    console.log('Game Data:', game);
  }, [state.status, state, game]);

  // 떠다니는 미니보드 초기화 (멀티플레이어 모드에서만)


  if (isLoading || gameStateLoading) return <div className="text-white text-center">Loading...</div>;
  if (error) return <div className="text-white text-center">Error: {error?.message}</div>;
  if (!state) return <div className="text-white text-center">No game state</div>;

  return (
    <div className={`game-screen flex items-center justify-center bg-[#1a1a2e]/90 ${
      isMinimized ? 'w-full h-full' : 'h-[88vh] w-[71vh]'
    }`}>

      <div className={`relative ${
        isMinimized ? 'w-full h-full' : 'h-[88vh] w-[71vh]'

      } bg-black/60 backdrop-blur-sm rounded-lg border border-white/10 shadow-2xl`}>
        {/* 사이드 정보 패널 */}
        <div className="absolute right-0 h-full py04 flex flex-col justify-between">
          {/* Hold */}
          <div className="bg-black/80 backdrop-blur-md p-4 rounded-lg border border-white/30">
            <div className="text-gray-200 text-sm mb-2 font-mono text-center">HOLD</div>
            <div className="w-20 h-20 border border-white/30 rounded-md overflow-hidden">
              {state.holdPiece && <MiniBoard {...holdPieceProps} />}
            </div>
          </div>

          {/* Next */}
          <div className="bg-black/80 backdrop-blur-md p-4 rounded-lg border border-white/30">
            <div className="text-gray-200 text-sm mb-2 font-mono">NEXT</div>
            <div className="w-20 h-20 border border-white/30 rounded-md overflow-hidden">
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


        {/* 메인 게임 보드 */}
        <div className="absolute inset-0 flex items-center justify-center">
          <div className={`game-grid ${
            isMinimized ? 'h-[80%] w-[35%]' : 'h-[90%] w-[61%]'
          } border-[1px] border-white/10 rounded-lg overflow-hidden bg-black/50 backdrop-blur-md`}>
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