import { useNavigate } from "react-router-dom";
import { useMutation, useQuery } from "convex/react";
import { api } from "../../convex/_generated/api";
import { GameList } from "../components/lobby/GameList";
import { Id } from "../../convex/_generated/dataModel";
import { useState } from 'react';

export default function Multiplayer() {
  const navigate = useNavigate();
  const createGame = useMutation(api.games.createGame);
  const games = useQuery(api.games.listGames);
  const [withAI, setWithAI] = useState(false);

  const handleCreateGame = async () => {
    try {
      const { gameId } = await createGame({
        playerId: localStorage.getItem('playerId') as Id<"players">,
        mode: "multi"
      });
      navigate(`/game/${gameId}`);
    } catch (error) {
      console.error('Failed to create game:', error);
    }
  };

  const handleJoinGame = (gameId: Id<"games">) => {
    navigate(`/game/${gameId}`);
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="grid md:grid-cols-3 gap-8">
        <div className="md:col-span-1">
          <div className="bg-gray-800 p-6 rounded-lg">
            <h2 className="text-2xl text-blue-400 mb-4">Create Game</h2>
            <div className="mb-4">
              <label className="flex items-center text-white">
                <input
                  type="checkbox"
                  checked={withAI}
                  onChange={(e) => setWithAI(e.target.checked)}
                  className="mr-2"
                />
                Play with AI
              </label>
            </div>
            <button 
              onClick={handleCreateGame}
              className="w-full bg-blue-500 text-white py-3 rounded hover:bg-blue-600"
            >
              Create New Game
            </button>
          </div>
        </div>
        <div className="md:col-span-2">
          <GameList games={games ?? []} onJoinGame={handleJoinGame} />
        </div>
      </div>
    </div>
  );
} 