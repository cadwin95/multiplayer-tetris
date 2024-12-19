import { useMutation } from "convex/react";
import { api } from "../../convex/_generated/api";
import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Id } from "../../convex/_generated/dataModel";

export default function Solo() {
  const navigate = useNavigate();
  const createGame = useMutation(api.games.createGame);
  const setReady = useMutation(api.games.setReady);

  useEffect(() => {
    const startSoloGame = async () => {
      const playerId = localStorage.getItem('playerId') as Id<"players">;
      if (!playerId) {
        navigate('/');
        return;
      }

      try {
        // 1. 게임 생성
        const { gameId } = await createGame({
          playerId,
          mode: 'solo'
        });

        // 2. 게임 ID를 플레이어에 연결
        await setReady({
          gameId,
          playerId
        });

        // 3. 게임 화면으로 이동
        navigate(`/game/${gameId}`);
      } catch (error) {
        console.error(error);
        navigate('/');
      }
    };

    startSoloGame();
  }, [createGame, setReady, navigate]);

  return <div className="text-white text-center">Starting game...</div>;
} 