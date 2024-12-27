// src/pages/Home.tsx
import { NicknameInput } from '../components/auth/NicknameInput';
import '../styles/tetris.css';
import { useState, useEffect } from 'react';
import { useGameCreation } from '../hooks/useGameCreation';
import { FloatingBoards } from '../components/game/FloatingBoards';

export default function Home() {
  const { startGame } = useGameCreation();
  const [hasNickname, setHasNickname] = useState(false);

  useEffect(() => {
    const nickname = localStorage.getItem('nickname');
    if (nickname) {
      setHasNickname(true);
    }
  }, []);

  const handleNicknameSubmit = (nickname: string) => {
    localStorage.setItem('nickname', nickname);
    setHasNickname(true);
  };

  if (!hasNickname) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-[#1a1a2e] to-[#16213e] flex items-center justify-center">
        <FloatingBoards />
        <div className="w-full max-w-md relative z-10">
          <NicknameInput onSubmit={handleNicknameSubmit} />
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-[#1a1a2e] to-[#16213e] flex items-center justify-center">
      <FloatingBoards />
      <div className="space-y-6 w-full max-w-md relative z-10">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">Tetris</h1>
          <p className="text-gray-400">
            Welcome, {localStorage.getItem('nickname')}!
          </p>
        </div>
        <div className="grid gap-4">
          <button 
            onClick={() => startGame('solo')} 
            className="menu-button bg-blue-500 hover:bg-blue-600"
          >
            Solo Play
          </button>
          <button 
            onClick={() => startGame('ai')} 
            className="menu-button bg-green-500 hover:bg-green-600"
          >
            Play vs AI
          </button>
          <button 
            onClick={() => startGame('multi')} 
            className="menu-button bg-purple-500 hover:bg-purple-600"
          >
            Multiplayer
          </button>
        </div>
      </div>
    </div>
  );
}