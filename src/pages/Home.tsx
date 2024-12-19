// src/pages/Home.tsx
import { Link } from 'react-router-dom';
import { NicknameInput } from '../components/auth/NicknameInput';
import '../styles/tetris.css';
import { useState, useEffect } from 'react';

export default function Home() {
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
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="w-full max-w-md">
          <NicknameInput onSubmit={handleNicknameSubmit} />
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 flex items-center justify-center">
      <div className="space-y-6 w-full max-w-md">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">Tetris</h1>
          <p className="text-gray-400">
            Welcome, {localStorage.getItem('nickname')}!
          </p>
        </div>
        <div className="grid gap-4">
          <Link 
            to="/solo" 
            className="menu-button bg-blue-500 hover:bg-blue-600"
          >
            Solo Play
          </Link>
          <Link 
            to="/ai" 
            className="menu-button bg-green-500 hover:bg-green-600"
          >
            Play vs AI
          </Link>
          <Link 
            to="/multiplayer" 
            className="menu-button bg-purple-500 hover:bg-purple-600"
          >
            Multiplayer
          </Link>
        </div>
      </div>
    </div>
  );
}