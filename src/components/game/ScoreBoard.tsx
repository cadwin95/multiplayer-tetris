interface ScoreBoardProps {
  score?: number;
  level?: number;
  lines?: number;
  players?: Array<{
    _id: string;
    name: string;
    score: number;
  }>;
}

export function ScoreBoard({ score, level, lines, players }: ScoreBoardProps) {
  if (players) {
    return (
      <div className="score-board">
        {players.map(player => (
          <div key={player._id} className="score-item">
            <label>{player.name}:</label>
            <span>{player.score}</span>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="score-board">
      <div className="score-item">
        <label>Score:</label>
        <span>{score}</span>
      </div>
      <div className="score-item">
        <label>Level:</label>
        <span>{level}</span>
      </div>
      <div className="score-item">
        <label>Lines:</label>
        <span>{lines}</span>
      </div>
    </div>
  );
}