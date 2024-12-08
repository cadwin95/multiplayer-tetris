interface ScoreBoardProps {
  players: Array<{
    name: string;
    score: number;
    _id: string;
  }>;
}

export function ScoreBoard({ players }: ScoreBoardProps) {
  return (
    <div className="bg-gray-900 p-4 rounded-lg min-w-[200px]">
      <h3 className="text-white text-xl mb-4 text-center">Scores</h3>
      <div className="space-y-2">
        {players.map(player => (
          <div key={player._id} className="flex justify-between items-center">
            <span className="text-white">{player.name}</span>
            <span className="text-blue-400">{player.score}</span>
          </div>
        ))}
      </div>
    </div>
  );
}