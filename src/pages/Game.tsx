import { useQuery, useMutation } from "convex/react";
import { api } from "../../convex/_generated/api";
import { useParams } from 'react-router-dom';
import { GameScreen } from '../components/game/GameScreen';
import { HumanGameScreen } from '../components/game/HumanGameScreen';
import { AIPlayer } from '../components/game/AIPlayer';
import { WaitingRoom } from '../components/lobby/WaitingRoom';
import { Id } from "../../convex/_generated/dataModel";
import { GameOver } from '../components/game/GameOver';
import { useNavigate } from 'react-router-dom';
import { useEffect, useState } from "react";
import { PIECE_ROTATIONS } from "../../convex/schema";
import { PieceType } from "../../convex/schema";
import { MiniBoard, MiniBoardProps } from '../components/game/MiniBoard';

interface FloatingBoard {
  id: number;
  x: number;
  y: number;
  speed: number;
  direction: number;
  piece: PieceType;
  rotation: number;
}

function FloatingBoards() {
  const [boards, setBoards] = useState<FloatingBoard[]>([]);
  
  // 초기 보드 생성 - 한 번만 실행
  useEffect(() => {
    const pieces = Object.keys(PIECE_ROTATIONS) as PieceType[];
    const newBoards: FloatingBoard[] = Array(12).fill(null).map((_, i) => ({
      id: i,
      x: Math.random() * 100,
      y: Math.random() * 100,
      speed: 0.2 + Math.random() * 0.3,
      direction: Math.random() * Math.PI * 2,
      piece: pieces[Math.floor(Math.random() * pieces.length)],
      rotation: Math.floor(Math.random() * 4)
    }));
    setBoards(newBoards);
  }, []); // 빈 의존성 배열

  // 자동 움직임 애니메이션
  useEffect(() => {
    const interval = setInterval(() => {
      setBoards(prevBoards => prevBoards.map(board => ({
        ...board,
        x: (board.x + Math.cos(board.direction) * board.speed) % 100,
        y: (board.y + Math.sin(board.direction) * board.speed) % 100,
        direction: board.direction + (Math.random() - 0.5) * 0.1
      })));
    }, 50);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none">
      {boards.map(board => {
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
            className="absolute w-16 h-16 transition-all duration-500 ease-linear opacity-40"
            style={{
              left: `${board.x}%`,
              top: `${board.y}%`,
              transform: `translate(-50%, -50%) rotate(${board.direction * 180 / Math.PI}deg)`,
              filter: 'drop-shadow(0 0 12px rgba(255, 255, 255, 0.4))',
            }}
          >
            <div className="bg-black/50 backdrop-blur-sm p-2 rounded-lg border border-white/20">
              <MiniBoard {...miniBoardProps} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default function Game() {
  const { gameId } = useParams<{ gameId: string }>();
  const playerId = localStorage.getItem('playerId') as Id<"players"> | null;
  const navigate = useNavigate();
  const game = useQuery(api.games.getGame, { gameId: gameId as Id<"games"> });
  const players = useQuery(api.games.getPlayers, { gameId: gameId as Id<"games"> });
  const setReady = useMutation(api.games.setReady);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (game && players) {
      const timer = setTimeout(() => {
        setIsLoading(false);
      }, 1000);

      return () => clearTimeout(timer);
    }
  }, [game, players]);

  if (!game || !players || !playerId) {
    return <div className="text-white text-center">Loading...</div>;
  }

  if (game.status === 'waiting') {
    return (
      <WaitingRoom 
        players={players.map(p => ({
          id: p._id,
          name: p.playerName,
          isReady: p.isReady || false
        }))}
        currentPlayerId={playerId}
        onReady={() => setReady({
          gameId: gameId as Id<"games">,
          playerId
        })}
      />
    );
  }

  const isAIGame = game.mode === 'ai';
  const aiPlayer = players.find(p => p.isAI);
  const currentPlayer = players.find(p => p._id === playerId);
  const otherPlayers = players.filter(p => p._id !== playerId);

  // 다른 플레이어 화면 위치 계산
  const getRandomPosition = (index: number) => {
    // 화면을 8개의 구역으로 나누고, 각 구역에 하나씩 배치
    const positions = [
      'top-4 left-4',        // 좌상단
      'top-4 right-4',       // 우상단
      'bottom-4 left-4',     // 좌하단
      'bottom-4 right-4',    // 우하단
      'top-4 left-1/3',      // 상단 왼쪽
      'top-4 right-1/3',     // 상단 오른쪽
      'bottom-4 left-1/3',   // 하단 왼쪽
      'bottom-4 right-1/3'   // 하단 오른쪽
    ];
    
    return positions[index % positions.length];
  };

  return (
    <div className="fixed inset-0 bg-gradient-to-b from-[#1a1a2e] to-[#16213e] overflow-hidden">
      <FloatingBoards />
      <div className="relative w-full h-full">
        {isLoading && (
          <div className="absolute inset-0 bg-black/90 backdrop-blur-md flex items-center justify-center z-50">
            <div className="text-white text-2xl font-mono">
              Ready...
            </div>
          </div>
        )}

        {isAIGame ? (
          <div className="flex h-full gap-16 px-16 -ml-16">
            {aiPlayer && (
              <div className="flex-1 h-full flex justify-end items-center">
                <AIPlayer 
                  gameId={gameId as Id<"games">} 
                  playerId={aiPlayer._id}
                  isLoading={isLoading}
                />
              </div>
            )}

            {currentPlayer && (
              <div className="flex-1 h-full flex justify-start items-center">
                <HumanGameScreen
                  gameId={gameId as Id<"games">}
                  playerId={currentPlayer._id}
                  isMinimized={false}
                  isLoading={isLoading}
                />
              </div>
            )}
          </div>
        ) : game.mode === 'multi' ? (
          // 멀티플레이어 레이아웃
          <>
            {/* 현재 플레이어의 게임 화면 - 중앙 배치 */}
            {currentPlayer && (
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-10">
                <HumanGameScreen
                  gameId={gameId as Id<"games">}
                  playerId={currentPlayer._id}
                  isMinimized={false}
                  isLoading={isLoading}
                />
              </div>
            )}

            {/* 다른 플레이어들의 게임 화면 - 주변 배치 */}
            {otherPlayers.map((player, index) => (
              <div
                key={player._id}
                className={`absolute ${getRandomPosition(index)} transform scale-50 opacity-60 transition-all duration-500 hover:opacity-90 hover:scale-[0.6] z-0 hover:z-20`}
                style={{
                  filter: 'blur(0.5px)',
                }}
              >
                <GameScreen
                  gameId={gameId as Id<"games">}
                  playerId={player._id}
                  isMinimized={true}
                  isLoading={isLoading}
                />
              </div>
            ))}
          </>
        ) : (
          // 솔로 모드 레이아웃 - 화면 중앙에 하나만 표시
          <div className="w-full h-full flex items-center justify-center">
            {currentPlayer && (
              <HumanGameScreen
                gameId={gameId as Id<"games">}
                playerId={currentPlayer._id}
                isMinimized={false}
                isLoading={isLoading}
              />
            )}
          </div>
        )}

        {/* GameOver 컴포넌트를 여기로 이동 */}
        {currentPlayer && game?.status === 'finished' && (
          <div className="absolute inset-0 bg-black/90 backdrop-blur-md flex items-center justify-center z-50">
            <GameOver 
              score={currentPlayer.score || 0}
              winnerId={game.winnerId}
              playerId={playerId}
              isAIGame={game.mode === 'ai'}
              aiScore={aiPlayer?.score}
              onPlayAgain={() => navigate('/')}
            />
          </div>
        )}
      </div>
    </div>
  );
}