import { Id } from "../../../convex/_generated/dataModel";
import { useGameState } from '../../hooks/useGameState';
import { useKeyboard } from '../../hooks/useKeyboard';
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
    currentPlayer,
    players,
    moveTetrominoe,
    rotateTetrominoe,
    hardDrop
  } = useGameState(gameId);

  // Keyboard handlers
  const keyboardHandlers = {
    onMoveLeft: () => moveTetrominoe({ gameId, playerId, direction: 'left' }),
    onMoveRight: () => moveTetrominoe({ gameId, playerId, direction: 'right' }),
    onMoveDown: () => moveTetrominoe({ gameId, playerId, direction: 'down' }),
    onRotate: () => rotateTetrominoe({ gameId, playerId }),
    onHardDrop: () => hardDrop({ gameId, playerId })
  };

  // Set up keyboard controls
  useKeyboard(game?.status === 'playing', keyboardHandlers);

  if (!game || !currentPlayer || !players) {
    return <div className="text-white text-center">Loading game state...</div>;
  }

  return (
    <div className="w-full min-h-screen bg-gray-900 p-8">
      <div className="max-w-7xl mx-auto flex flex-col items-center gap-8">
        {/* Game Boards Container */}
        <div className="flex justify-center gap-8 p-8 bg-gray-800/80 rounded-lg shadow-lg w-full">
          {players.map((player) => (
            <div 
              key={player._id} 
              className="flex flex-col items-center bg-gray-800 p-6 rounded-lg shadow-md"
            >
              <h2 className="text-white text-xl font-bold mb-4">
                {player._id === playerId ? `${player.name} (You)` : player.name}
              </h2>
              
              <div className="border border-gray-700 rounded-lg overflow-hidden">
                <GameBoard
                  board={player.board}
                  currentPiece={player.currentPiece}
                  position={player.position}
                />
              </div>
              
              {player._id === playerId && (
                <div className="mt-4">
                  <NextPiece piece={player.nextPiece} />
                </div>
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