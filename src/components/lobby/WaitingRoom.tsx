import { Id } from "../../../convex/_generated/dataModel";

interface WaitingRoomProps {
  players: Array<{
    id: Id<"players">;
    name: string;
    isReady: boolean;
  }>;
  onReady: () => void;
  currentPlayerId: Id<"players">;
}

export function WaitingRoom({ players, onReady, currentPlayerId }: WaitingRoomProps) {
  const currentPlayer = players.find(p => p.id === currentPlayerId);
  const allReady = players.every(p => p.isReady);

  return (
    <div className="bg-gray-800 p-6 rounded-lg">
      <h2 className="text-2xl text-blue-400 mb-4">Waiting Room</h2>
      <div className="space-y-4 mb-6">
        {players.map(player => (
          <div 
            key={player.id} 
            className="flex items-center justify-between"
          >
            <span className="text-white">
              {player.name} {player.id === currentPlayerId && "(You)"}
            </span>
            <span className={`px-2 py-1 rounded ${
              player.isReady ? 'bg-green-500' : 'bg-red-500'
            }`}>
              {player.isReady ? 'Ready' : 'Not Ready'}
            </span>
          </div>
        ))}
      </div>
      {!currentPlayer?.isReady && (
        <button
          onClick={onReady}
          className="w-full bg-green-500 text-white py-2 rounded hover:bg-green-600"
        >
          Ready
        </button>
      )}
      {allReady && (
        <div className="text-green-400 text-center mt-4">
          All players ready! Game starting...
        </div>
      )}
    </div>
  );
} 