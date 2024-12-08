import { useQuery, useMutation } from "convex/react";
import { api } from "../../convex/_generated/api";
import { Id } from "../../convex/_generated/dataModel";
import { ScoreBoard } from '../components/game/ScoreBoard';
import { GameScreen } from '../components/game/GameScreen';
import { useParams } from 'react-router-dom';
import { useEffect, useState } from "react";

export default function Game() {
  // URL에서 gameId 파라미터 가져오기
  const { gameId: urlGameId } = useParams();
  const [playerId, setPlayerId] = useState<string | null>(null);

  // localStorage에서 playerId만 가져오기
  useEffect(() => {
    setPlayerId(localStorage.getItem('playerId'));
  }, []);

  const startGame = useMutation(api.games.startGame);
  const setReady = useMutation(api.games.setReady);

  const game = useQuery(
    api.games.getGame, 
    urlGameId ? { gameId: urlGameId as Id<"games"> } : "skip"
  );
  
  const currentPlayer = useQuery(
    api.games.getPlayer, 
    playerId ? { playerId: playerId as Id<"players"> } : "skip"
  );

  const players = useQuery(
    api.games.getPlayers,
    urlGameId ? { gameId: urlGameId as Id<"games"> } : "skip"
  );

  if (!urlGameId || !playerId || !game || !currentPlayer || !players) {
    return <div className="text-white text-center">Loading...</div>;
  }

  const handleReady = async () => {
    try {
      await setReady({
        gameId: urlGameId as Id<"games">,
        playerId: playerId as Id<"players">
      });
    } catch (error) {
      console.error("Failed to set ready:", error);
    }
  };

  const handleStartGame = async () => {
    try {
      await startGame({ gameId: urlGameId as Id<"games"> });
    } catch (error) {
      console.error("Failed to start game:", error);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 py-8">
      <div className="max-w-7xl mx-auto px-4">
        {/* ... 나머지 JSX는 동일 ... */}
        {game.status === 'playing' ? (
          <GameScreen 
            gameId={urlGameId as Id<"games">} 
            playerId={playerId as Id<"players">}
          />
        ) : (
          <div className="flex justify-center items-start gap-8">
            <div className="flex flex-col gap-6">
              <ScoreBoard players={players.map(player => ({
                _id: player._id,
                name: player._id === playerId ? `${player.name} (You)` : player.name,
                score: player.score
              }))} />
              
              {game.status === "waiting" && !currentPlayer.isReady && (
                <button 
                  className="px-8 py-3 bg-green-500 text-white rounded-lg font-bold"
                  onClick={handleReady}
                >
                  Ready
                </button>
              )}

              {game.status === "waiting" && currentPlayer.isReady && (
                <button 
                  className="px-8 py-3 bg-blue-500 text-white rounded-lg font-bold"
                  onClick={handleStartGame}
                >
                  Start Game
                </button>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}