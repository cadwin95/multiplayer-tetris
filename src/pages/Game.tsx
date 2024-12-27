import { useQuery, useMutation } from "convex/react";
import { api } from "../../convex/_generated/api";
import { useParams } from 'react-router-dom';
import { GameScreen } from '../components/game/GameScreen';
import { AIPlayer } from '../components/game/AIPlayer';
import { WaitingRoom } from '../components/lobby/WaitingRoom';
import { Id } from "../../convex/_generated/dataModel";

export default function Game() {
  const { gameId } = useParams<{ gameId: string }>();
  const playerId = localStorage.getItem('playerId') as Id<"players"> | null;
  
  const game = useQuery(api.games.getGame, { 
    gameId: gameId as Id<"games"> 
  });
  const players = useQuery(api.games.getPlayers, { 
    gameId: gameId as Id<"games"> 
  });
  const setReady = useMutation(api.games.setReady);

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
      <div className="relative w-full h-full">
        {/* AI 모드일 때 좌우 분할 레이아웃 */}
        {isAIGame ? (
          <div className="flex w-full h-full">
            {/* AI 플레이어 - 좌측 */}
            <div className="w-1/2 h-full flex items-center justify-center relative">
              {aiPlayer && (
                <>
                  <div className="hidden">
                    <AIPlayer 
                      gameId={gameId as Id<"games">} 
                      playerId={aiPlayer._id} 
                    />
                  </div>
                  <GameScreen
                    gameId={gameId as Id<"games">}
                    playerId={aiPlayer._id}
                    isMinimized={false}
                  />
                </>
              )}
              <div className="absolute top-4 left-1/2 -translate-x-1/2 text-white/50 text-xl font-mono">
                AI Player
              </div>
            </div>

            {/* 사람 플레이어 - 우측 */}
            <div className="w-1/2 h-full flex items-center justify-center relative">
              {currentPlayer && (
                <>
                  <GameScreen
                    gameId={gameId as Id<"games">}
                    playerId={currentPlayer._id}
                    isMinimized={false}
                  />
                  <div className="absolute top-4 left-1/2 -translate-x-1/2 text-white/50 text-xl font-mono">
                    Human Player
                  </div>
                </>
              )}
            </div>
          </div>
        ) : game.mode === 'multi' ? (
          // 멀티플레이어 레이아웃
          <>
            {/* 현재 플레이어의 게임 화면 - 중앙 배치 */}
            {currentPlayer && (
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-10">
                <GameScreen
                  gameId={gameId as Id<"games">}
                  playerId={currentPlayer._id}
                  isMinimized={false}
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
                />
              </div>
            ))}
          </>
        ) : (
          // 솔로 모드 레이아웃 - 화면 중앙에 하나만 표시
          <div className="w-full h-full flex items-center justify-center">
            {currentPlayer && (
              <GameScreen
                gameId={gameId as Id<"games">}
                playerId={currentPlayer._id}
                isMinimized={false}
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
}