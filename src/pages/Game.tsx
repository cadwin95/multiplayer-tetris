import { useQuery, useMutation } from "convex/react";
import { api } from "../../convex/_generated/api";
import { Id } from "../../convex/_generated/dataModel";
import { useParams } from 'react-router-dom';
import { useEffect } from 'react';
import { WaitingRoom } from '../components/lobby/WaitingRoom';
import { AIPlayer } from '../components/game/AIPlayer';
import { GameScreen } from '../components/game/GameScreen';

export default function Game() {
  const { gameId: urlGameId } = useParams();
  const playerId = localStorage.getItem('playerId') as Id<"players">;
  
  const game = useQuery(api.games.getGame, { gameId: urlGameId as Id<"games"> });
  const players = useQuery(api.games.getPlayers, { gameId: urlGameId as Id<"games"> });
  
  const setReady = useMutation(api.games.setReady);
  const startGameAfterDelay = useMutation(api.games.startGameAfterDelay);

  useEffect(() => {
    if (game?.status === "ready") {
      const timer = setTimeout(async () => {
        await startGameAfterDelay({ gameId: urlGameId as Id<"games"> });
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [game?.status, startGameAfterDelay, urlGameId]);

  // AI 플레이어 확인 및 찾기
  const aiPlayer = players?.find(p => p.isAI);
  const humanPlayer = players?.find(p => p._id === playerId);

  if (!game || !players) {
    return <div className="text-white text-center">Loading...</div>;
  }

  if (game.status === 'waiting') {
    return (
      <div className="container mx-auto px-4 py-8">
        <WaitingRoom 
          players={players.map(p => ({
            id: p._id,
            name: p.playerName,
            isReady: p.isReady
          }))}
          currentPlayerId={playerId}
          onReady={() => setReady({
            gameId: urlGameId as Id<"games">,
            playerId
          })}
        />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 py-8">
      <div className="container mx-auto px-4">
        {/* AI 컨트롤러 */}
        {aiPlayer && <AIPlayer gameId={urlGameId as Id<"games">} playerId={aiPlayer._id} />}
        
        <div className="flex gap-8 justify-center items-start">
          {/* AI 플레이어 미니 화면 */}
          {aiPlayer && (
            <div className="mini-game-container">
              <div className="mb-2">
                <h2 className="text-lg text-white font-bold">
                  {aiPlayer.playerName} (AI)
                </h2>
              </div>
              <div className="mini-game-board">
                <GameScreen 
                  gameId={urlGameId as Id<"games">} 
                  playerId={aiPlayer._id}
                  isMinimized={true}
                />
              </div>
            </div>
          )}

          {/* 메인 플레이어 화면 */}
          <div className="main-game-container">
            <div className="mb-4">
              <h2 className="text-2xl text-white font-bold">
                {humanPlayer?.playerName} (You)
              </h2>
            </div>
            <div className="main-game-board">
              <GameScreen 
                gameId={urlGameId as Id<"games">} 
                playerId={playerId}
                isMinimized={false}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}