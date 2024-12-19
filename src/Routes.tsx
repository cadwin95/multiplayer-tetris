import { Routes as RouterRoutes, Route } from 'react-router-dom';
import Home from './pages/Home';
import Game from './pages/Game';
import Solo from './pages/Solo';
import Multiplayer from './pages/Multiplayer';
import About from './pages/About';

export default function Routes() {
  return (
    <RouterRoutes>
      <Route path="/" element={<Home />} />
      <Route path="/game/:gameId" element={<Game />} />
      <Route path="/solo" element={<Solo />} />
      <Route path="/multiplayer" element={<Multiplayer />} />
      <Route path="/about" element={<About />} />
    </RouterRoutes>
  );
} 