import { useState } from 'react';

export default function Config() {
  const [das, setDas] = useState(100);
  const [arr, setArr] = useState(0);
  const [volume, setVolume] = useState(50);

  const handleSave = () => {
    localStorage.setItem('tetris-config', JSON.stringify({
      das, arr, volume
    }));
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-2xl mx-auto bg-gray-800 rounded-lg p-6">
        <h1 className="text-3xl text-white mb-8">Game Configuration</h1>
        
        <div className="space-y-6">
          <div className="config-item">
            <label className="text-white">DAS (Delayed Auto Shift)</label>
            <input 
              type="range" 
              min="0" 
              max="200" 
              value={das}
              onChange={(e) => setDas(Number(e.target.value))}
              className="w-full"
            />
            <span className="text-white">{das}ms</span>
          </div>

          <div className="config-item">
            <label className="text-white">ARR (Auto Repeat Rate)</label>
            <input 
              type="range" 
              min="0" 
              max="100" 
              value={arr}
              onChange={(e) => setArr(Number(e.target.value))}
              className="w-full"
            />
            <span className="text-white">{arr}ms</span>
          </div>

          <div className="config-item">
            <label className="text-white">Volume</label>
            <input 
              type="range" 
              min="0" 
              max="100" 
              value={volume}
              onChange={(e) => setVolume(Number(e.target.value))}
              className="w-full"
            />
            <span className="text-white">{volume}%</span>
          </div>

          <button 
            onClick={handleSave}
            className="w-full bg-blue-500 text-white py-2 rounded hover:bg-blue-600"
          >
            Save Configuration
          </button>
        </div>
      </div>
    </div>
  );
} 