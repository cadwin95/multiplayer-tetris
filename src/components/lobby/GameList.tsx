// src/components/lobby/GameList.tsx
import { useQuery, useMutation } from "convex/react";
import { api } from "../../../convex/_generated/api";
import { Id } from "../../../convex/_generated/dataModel";
import { useNavigate } from 'react-router-dom';  // 추가

function PlayerName({ playerId }: { playerId: Id<"players"> }) {
  const playerInfo = useQuery(api.games.getPlayer, { playerId });
  return (
    <div className="flex items-center space-x-2 text-gray-300">
      <div className="w-2 h-2 rounded-full bg-green-500"></div>
      <span>{playerInfo?.name}</span>
    </div>
  );
}

export function GameList() {
  const games = useQuery(api.games.listGames);
  const joinGame = useMutation(api.games.joinGame);
  const navigate = useNavigate();  // 추가

  const handleJoinGame = async (gameId: string) => {
    const playerName = prompt("Enter your name:");
    if (!playerName) return;

    try {
      const playerId = await joinGame({
        gameId: gameId as Id<"games">,
        playerName: playerName.trim()
      });
      localStorage.setItem('gameId', gameId);
      localStorage.setItem('playerId', playerId);
      navigate(`/game/${gameId}`);
    } catch {
      alert("Failed to join game");
    }
  };

  if (!games) return <div className="text-gray-400">Loading available games...</div>;

  return (
    <div>
      <h2 className="text-2xl text-[#ff0055] mb-6">Available Games</h2>
      <div className="space-y-4">  {/* 게임 카드 간 여백 */}
        {games.map((game) => (
          <div key={game._id} 
               className="bg-[#1a1a2e] p-6 rounded border border-gray-800">
            <div className="flex justify-between items-center mb-4">
              <span className="text-lg text-gray-300">Game #{game._id.slice(-8)}</span>
              <span className="px-3 py-1 bg-[#ff0055]/10 text-[#ff0055] rounded">
                {game.status}
              </span>
            </div>
            
            <div className="mb-4">
              <p className="text-gray-400 mb-2">Players ({game.players.length})</p>
              <div className="space-y-2 pl-4">  {/* 플레이어 목록 들여쓰기 */}
                {game.players.map((playerId: Id<"players">) => (
                  <PlayerName key={playerId} playerId={playerId} />
                ))}
              </div>
            </div>

            {game.status === "waiting" && (
              <button 
                onClick={() => handleJoinGame(game._id)}
                className="w-full bg-[#00ff00] text-black font-bold py-3 rounded"
              >
                Join Game
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}