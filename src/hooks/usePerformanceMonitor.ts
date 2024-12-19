import { useState, useCallback } from 'react';

interface PerformanceMetrics {
  averageLatency: number;
  sampleCount: number;
}

export function usePerformanceMonitor() {
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    averageLatency: 0,
    sampleCount: 0
  });

  const [lastActionTime, setLastActionTime] = useState<number | null>(null);

  const measureLatency = useCallback(() => {
    if (lastActionTime) {
      const currentTime = performance.now();
      const latency = currentTime - lastActionTime;

      setMetrics(prev => ({
        averageLatency: (prev.averageLatency * prev.sampleCount + latency) / (prev.sampleCount + 1),
        sampleCount: prev.sampleCount + 1
      }));
    }
  }, [lastActionTime]);

  const startMeasurement = useCallback(() => {
    setLastActionTime(performance.now());
  }, []);

  return {
    metrics,
    startMeasurement,
    measureLatency
  };
} 