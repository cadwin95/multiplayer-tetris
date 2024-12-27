import { useNavigate } from 'react-router-dom';
import { Id } from "../../../convex/_generated/dataModel";

interface GameOverProps {
  score: number;
  winnerId?: Id<"players">;
  playerId: Id<"players">;
  isAIGame?: boolean;
  aiScore?: number;
  onPlayAgain?: () => void;
}

export function GameOver({ score, winnerId, playerId, isAIGame, aiScore, onPlayAgain }: GameOverProps) {
  const navigate = useNavigate();
  
  const isWinner = winnerId === playerId;
  const gameResult = isAIGame 
    ? (score > (aiScore || 0) ? '승리!' : score === (aiScore || 0) ? '무승부!' : '패배!')
    : (isWinner ? '승리!' : '게임 오버!');

  return (
    <div className="text-center text-white">
      <h2 className="text-4xl font-bold mb-4">{gameResult}</h2>
      {isAIGame && (
        <div className="mb-6 space-y-2">
          <p className="text-xl">플레이어 점수: {score.toLocaleString()}</p>
          <p className="text-xl">AI 점수: {aiScore?.toLocaleString()}</p>
        </div>
      )}
      {!isAIGame && (
        <p className="text-2xl mb-6">점수: {score.toLocaleString()}</p>
      )}
      <div className="space-x-4">
        <button
          onClick={() => navigate('/')}
          className="px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg text-white font-bold transition-colors"
        >
          홈으로
        </button>
        {onPlayAgain && (
          <button
            onClick={onPlayAgain}
            className="px-6 py-3 bg-green-600 hover:bg-green-700 rounded-lg text-white font-bold transition-colors"
          >
            다시하기
          </button>
        )}
      </div>
    </div>
  );
} 