// src/pages/Home.tsx
import { CreateGame } from "../components/lobby/CreateGame";
import { GameList } from "../components/lobby/GameList";
import '../styles/tetris.css';

export default function Home() {
  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      <div className="container mx-auto px-4 md:px-8">
        {/* 로고 섹션 */}
        <div className="text-center pt-16 mb-16">
          <h1 className="tetris-logo text-7xl font-extrabold mb-4">
            <span>T</span>
            <span>E</span>
            <span>T</span>
            <span>R</span>
            <span>I</span>
            <span>S</span>
          </h1>
        </div>

        {/* 메인 컨텐츠 */}
        <div className="grid md:grid-cols-3 gap-8">
          {/* Create Game 섹션 */}
          <div className="md:col-span-1">
            <div className="game-card bg-gray-800/50 p-8 rounded-xl border-2 border-blue-500/30">
              <h2 className="text-2xl text-blue-400 mb-6 font-bold">Create Game</h2>
              <CreateGame />
            </div>
          </div>

          {/* Game List 섹션 */}
          <div className="md:col-span-2">
            <div className="game-card bg-gray-800/50 p-8 rounded-xl border-2 border-purple-500/30">
              <GameList />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}