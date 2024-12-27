import { useNavigate } from 'react-router-dom';
import { useMutation } from "convex/react";
import { api } from "../../../convex/_generated/api";
import { Id } from "../../../convex/_generated/dataModel";

interface GameOverProps {
  score: number;
  winnerId?: Id<"players">;
  playerId: Id<"players">;
  onPlayAgain: () => void;
}

export function GameOver({ score, winnerId, playerId, onPlayAgain }: GameOverProps) {
  const navigate = useNavigate();
  const createPlayer = useMutation(api.games.createPlayer);

  const handleRestart = async () => {
    try {
      const nickname = localStorage.getItem('nickname');
      localStorage.clear();
      
      if (nickname) {
        localStorage.setItem('nickname', nickname);
        const newPlayerId = await createPlayer({
          playerName: nickname
        });
        localStorage.setItem('playerId', newPlayerId);
      }

      onPlayAgain();
    } catch (error) {
      console.error('Failed to restart game:', error);
      navigate('/', { replace: true });
    }
  };

  const isWinner = winnerId === playerId;

  return (
    <div className="h-full flex items-center justify-center bg-black/80">
      <div className="bg-gray-800 p-8 rounded-lg text-center">
        <h2 className={`text-4xl mb-4 ${isWinner ? 'text-green-500' : 'text-red-500'}`}>
          {isWinner ? 'You Won!' : 'Game Over'}
        </h2>
        <p className="text-xl text-white mb-2">Final Score: {score}</p>
        <p className="text-lg text-gray-400 mb-6">
          {isWinner ? 'Congratulations!' : 'Better luck next time!'}
        </p>
        <div className="space-y-4">
          <button
            onClick={handleRestart}
            className="w-full px-6 py-3 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Play Again
          </button>
          <button
            onClick={() => navigate('/')}
            className="w-full px-6 py-3 bg-gray-600 text-white rounded hover:bg-gray-700"
          >
            Back to Main
          </button>
        </div>
      </div>
    </div>
  );
} 