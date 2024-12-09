import { Id } from "../../../convex/_generated/dataModel";
import { useGameState } from '../../hooks/useGameState';
import { useKeyboard } from '../../hooks/useKeyboard';
import { useGameTimer } from '../../hooks/useGameTimer';
import { GameBoard } from './GameBoard';
import { NextPiece } from './NextPiece';
import { ScoreBoard } from './ScoreBoard';


interface GameScreenProps {
  gameId: Id<"games">;
  playerId: Id<"players">;
}

export function GameScreen({ gameId, playerId }: GameScreenProps) {
  const {
    game,
    players,
    gameState,
    moveTetrominoe,
    rotateTetrominoe,
    hardDrop
  } = useGameState(gameId);

  // 키보드 핸들러 - 이제 로컬 상태를 즉시 업데이트
  const keyboardHandlers = {
    onMoveLeft: () => moveTetrominoe('left'),
    onMoveRight: () => moveTetrominoe('right'),
    onMoveDown: () => moveTetrominoe('down'),
    onRotate: () => rotateTetrominoe(),
    onHardDrop: () => hardDrop({ gameId, playerId })
  };

  useKeyboard(game?.status === 'playing', keyboardHandlers);
  
  // 자동 하강 타이머
  useGameTimer(
    game?.status === 'playing',
    gameId,
    playerId,
    gameState?.level || 1
  );

  if (!game || !gameState || !players) {
    return <div className="text-white text-center">Loading...</div>;
  }

  return (
    <div className="w-full min-h-screen bg-gray-900 p-8">
      <div className="max-w-7xl mx-auto flex flex-col items-center gap-8">
        <div className="flex justify-center gap-8 p-8 bg-gray-800/80 rounded-lg shadow-lg w-full">
          {players.map((player) => (
            <div key={player._id} 
                 className={`flex flex-col items-center bg-gray-800 p-6 rounded-lg shadow-md
                            ${!gameState.isValid ? 'opacity-75' : ''}`}>
              <h2 className="text-white text-xl font-bold mb-4">
                {player._id === playerId ? `${player.name} (You)` : player.name}
              </h2>
              
              <GameBoard
                board={gameState.board}
                currentPiece={gameState.currentPiece}
                position={gameState.position}
              />
              
              {player._id === playerId && (
                <NextPiece piece={gameState.nextPiece} />
              )}
            </div>
          ))}
        </div>
        
        {/* Controls and Score Panel */}
        <div className="flex gap-8 justify-center w-full">
          <ScoreBoard players={players} />
          
          <div className="bg-gray-800 p-4 rounded-lg shadow-md min-w-[200px]">
            <h3 className="text-white text-lg font-bold mb-2">
              Controls
            </h3>
            <ul className="text-gray-300 space-y-1">
              <li>← → : Move</li>
              <li>↓ : Soft Drop</li>
              <li>↑ : Rotate</li>
              <li>Space : Hard Drop</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}