import { useQuery, useMutation } from "convex/react";
import { api } from "../../convex/_generated/api";
import { Id } from "../../convex/_generated/dataModel";
import { useParams, useNavigate } from 'react-router-dom';
import { useEffect } from 'react';
import { GameScreen } from '../components/game/GameScreen';

export default function GamePage() {
  const navigate = useNavigate();
  const { gameId } = useParams();
  const playerId = localStorage.getItem('playerId') as Id<"players">;

  // Queries
  const game = useQuery(api.games.getGame, { gameId: gameId as Id<"games"> });
  const players = useQuery(api.games.getPlayers, { gameId: gameId as Id<"games"> });

  if (!game || !players) {
    return <div className="text-white text-center">Loading...</div>;
  }

  const currentPlayer = players.find(p => p._id === playerId);
  const otherPlayers = players.filter(p => p._id !== playerId);

  return (
    <div className="min-h-screen bg-gray-900 py-8">
      <div className="container mx-auto px-4">
        <div className="flex gap-8 justify-center items-start">
          {/* 다른 플레이어들의 미니 화면 */}
          {otherPlayers.map(player => (
            <div key={player._id} className="mini-game-container">
              <div className="mb-2">
                <h2 className="text-lg text-white font-bold">
                  {player.playerName}
                </h2>
              </div>
              <div className="mini-game-board">
                <GameScreen 
                  gameId={gameId as Id<"games">} 
                  playerId={player._id}
                  isMinimized={true}
                />
              </div>
            </div>
          ))}

          {/* 메인 플레이어 화면 */}
          <div className="main-game-container">
            <div className="mb-4">
              <h2 className="text-2xl text-white font-bold">
                {currentPlayer?.playerName} (You)
              </h2>
            </div>
            <div className="main-game-board">
              <GameScreen 
                gameId={gameId as Id<"games">} 
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