// src/App.tsx
import { MainNav } from './components/layout/MainNav';
import { Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import Solo from './pages/Solo';
import AIGame from './pages/AIGame';
import Multiplayer from './pages/Multiplayer';
import Game from './pages/Game';

function App() {
  return (
    <div className="min-h-screen bg-gray-900">
      <MainNav />
      <main className="pt-32">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/solo" element={<Solo />} />
          <Route path="/ai" element={<AIGame />} />
          <Route path="/multiplayer" element={<Multiplayer />} />
          <Route path="/game/:gameId" element={<Game />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;