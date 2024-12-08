// src/App.tsx
import { Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import Game from './pages/Game';

function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/game/:gameId" element={<Game />} />
    </Routes>
  );
}

export default App;