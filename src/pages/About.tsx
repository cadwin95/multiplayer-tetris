export default function About() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      <div className="container mx-auto px-4 py-16">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-white mb-8">About Tetris</h1>
          <div className="text-gray-300 space-y-4">
            <p>A modern multiplayer Tetris game built with React and Convex.</p>
            <p>Play solo or challenge your friends in real-time multiplayer matches.</p>
          </div>
        </div>
      </div>
    </div>
  );
} 