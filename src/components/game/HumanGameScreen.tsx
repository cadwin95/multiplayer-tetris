import { GameScreen } from './GameScreen';
import { useKeyboard } from '../../hooks/useKeyboard';
import { useGameState } from '../../hooks/useGameState';
import { Id } from "../../../convex/_generated/dataModel";

interface HumanGameScreenProps {
  gameId: Id<"games">;
  playerId: Id<"players">;
  isMinimized?: boolean;
  isLoading?: boolean;
}

export function HumanGameScreen({ gameId, playerId, isMinimized = false }: HumanGameScreenProps) {
  const { move } = useGameState(gameId, playerId);

  useKeyboard({
    isEnabled: !isMinimized,
    onMoveLeft: () => move('left'),
    onMoveRight: () => move('right'),
    onMoveDown: () => move('down'),
    onRotate: () => move('rotate'),
    onHardDrop: () => move('hardDrop'),
    onHold: () => move('hold')
  });

  return (
    <div className="relative">
      <GameScreen
        gameId={gameId}
        playerId={playerId}
        isMinimized={isMinimized}
      />
      {!isMinimized && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 text-white text-center text-xl font-mono z-10 border border-white p-2 rounded-lg">
          Human Player
        </div>
      )}
    </div>
  );
} 