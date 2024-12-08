import { useState } from "react";
import { useMutation } from "convex/react";
import { api } from "../../../convex/_generated/api";
import { useNavigate } from 'react-router-dom';

export function CreateGame() {
  const [playerName, setPlayerName] = useState("");
  const createGame = useMutation(api.games.createGame);
  const navigate = useNavigate();
  const handleCreateGame = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!playerName.trim()) return;

    try {
      const { gameId, playerId } = await createGame({
        playerName: playerName.trim(),
      });
      
      localStorage.setItem('gameId', gameId);
      localStorage.setItem('playerId', playerId);
      
      // URL에 gameId와 playerId 모두 포함하여 이동
      navigate(`/game/${gameId}`);
    } catch (error) {
      console.error("Failed to create game:", error);
    }
  };

  return (
    <form onSubmit={handleCreateGame} className="space-y-4">  {/* 폼 요소 간 여백 */}
      <input
        type="text"
        value={playerName}
        onChange={(e) => setPlayerName(e.target.value)}
        placeholder="Enter your name"
        className="w-full px-4 py-3 bg-[#1a1a2e] border border-gray-700 rounded"
        required
      />
      <button 
        type="submit"
        className="w-full bg-[#00f0ff] text-black font-bold py-3 rounded"
      >
        Start Game
      </button>
    </form>
  );
}