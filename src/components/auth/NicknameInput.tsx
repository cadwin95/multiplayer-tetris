import { useState } from 'react';
import { useMutation } from 'convex/react';
import { api } from '../../../convex/_generated/api';

interface NicknameInputProps {
  onSubmit: (nickname: string) => void;
}

export function NicknameInput({ onSubmit }: NicknameInputProps) {
  const [nickname, setNickname] = useState('');
  const createPlayer = useMutation(api.games.createPlayer);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!nickname.trim()) return;

    try {
      // 1. 플레이어 생성
      const playerId = await createPlayer({
        playerName: nickname.trim()
      });

      // 2. 플레이어 ID를 localStorage에 저장
      localStorage.setItem('playerId', playerId);
      localStorage.setItem('nickname', nickname.trim());

      // 3. 부모 컴포넌트에 알림
      onSubmit(nickname);
    } catch (error) {
      console.error('Failed to create player:', error);
      alert('Failed to create player. Please try again.');
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 flex items-center justify-center">
      <form 
        onSubmit={handleSubmit}
        className="bg-gray-800 p-8 rounded-lg shadow-lg w-full max-w-md"
      >
        <h2 className="text-2xl text-white mb-6 text-center">Enter Your Nickname</h2>
        <input
          type="text"
          value={nickname}
          onChange={(e) => setNickname(e.target.value)}
          className="w-full px-4 py-2 bg-gray-700 text-white rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder="Your nickname..."
          maxLength={20}
          required
        />
        <button
          type="submit"
          className="w-full mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
          disabled={!nickname.trim()}
        >
          Start Playing
        </button>
      </form>
    </div>
  );
} 