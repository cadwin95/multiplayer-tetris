import { Routes as RouterRoutes, Route } from 'react-router-dom';
import Home from './pages/Home';
import Game from './pages/Game';

export default function Routes() {
  return (
    <RouterRoutes>
      <Route path="/" element={<Home />} />
      <Route path="/game/:gameId" element={<Game />} />
    </RouterRoutes>
  );
} 