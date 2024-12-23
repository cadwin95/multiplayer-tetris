// src/App.tsx
import { MainNav } from './components/layout/MainNav';
import { Routes, Route, useNavigate } from 'react-router-dom';
import GamePage from './pages/GamePage';
import { useMutation } from "convex/react";
import { api } from "../convex/_generated/api";
import { useEffect } from 'react';
import { Id } from "../convex/_generated/dataModel";

function CreateGame() {
  const navigate = useNavigate();
  const createGame = useMutation(api.games.createGame);
  const setReady = useMutation(api.games.setReady);
  const createPlayer = useMutation(api.games.createPlayer);

  useEffect(() => {
    const initGame = async () => {
      try {
        // 1. 플레이어 생성 또는 기존 플레이어 ID 가져오기
        let playerId: Id<"players">;
        const storedPlayerId = localStorage.getItem('playerId');
        
        if (!storedPlayerId) {
          // 새 플레이어 생성
          const playerName = `Player${Math.floor(Math.random() * 1000)}`;
          playerId = await createPlayer({ playerName });
          console.log('Created new player:', playerId);
          localStorage.setItem('playerId', playerId);
        } else {
          // 기존 플레이어 ID 사용
          playerId = storedPlayerId as Id<"players">;
          console.log('Using existing player:', playerId);
        }

        // 2. 게임 생성
        console.log('Creating game with playerId:', playerId);
        const result = await createGame({
          playerId,
          mode: 'solo'
        });
        console.log('Game creation result:', result);

        if (!result?.gameId) {
          throw new Error('Failed to create game: Invalid game ID');
        }

        // 3. 플레이어 ready 상태 설정
        console.log('Setting player ready state');
        await setReady({
          gameId: result.gameId,
          playerId
        });

        // 4. 게임 페이지로 이동
        console.log('Navigating to game:', result.gameId);
        navigate(`/game/${result.gameId}`);
      } catch (error) {
        console.error('Failed to initialize game:', error);
        // 에러가 발생한 경우 playerId를 제거하고 페이지를 새로고침
        localStorage.removeItem('playerId');
        window.location.reload();
      }
    };

    initGame();
  }, [createGame, setReady, createPlayer, navigate]);

  return <div className="text-white text-center">Creating game...</div>;
}

function App() {
  return (
    <div className="min-h-screen bg-gray-900">
      <MainNav />
      <main className="pt-32">
        <Routes>
          <Route path="/" element={<CreateGame />} />
          <Route path="/game/:gameId" element={<GamePage />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;