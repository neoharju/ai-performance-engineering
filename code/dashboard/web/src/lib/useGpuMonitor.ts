'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { getGpuInfo } from './api';

export interface GpuMetrics {
  gpu_name: string;
  gpu_count: number;
  temperature_gpu: number;
  temperature_memory: number;
  power_draw_w: number;
  power_limit_w: number;
  memory_used_gb: number;
  memory_total_gb: number;
  utilization_gpu: number;
  utilization_memory: number;
  clock_graphics_mhz: number;
  clock_memory_mhz: number;
  fan_speed: number;
  persistence_mode: boolean;
  pstate: string;
  timestamp?: number;
}

interface UseGpuMonitorOptions {
  interval?: number;  // Polling interval in ms (default: 5000)
  enabled?: boolean;  // Whether to enable polling
}

interface UseGpuMonitorResult {
  data: GpuMetrics | null;
  history: GpuMetrics[];
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  isPolling: boolean;
  setPolling: (enabled: boolean) => void;
}

export function useGpuMonitor(options: UseGpuMonitorOptions = {}): UseGpuMonitorResult {
  const { interval = 5000, enabled = true } = options;
  
  const [data, setData] = useState<GpuMetrics | null>(null);
  const [history, setHistory] = useState<GpuMetrics[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isPolling, setIsPolling] = useState(enabled);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const fetchGpuData = useCallback(async () => {
    try {
      const result = await getGpuInfo() as GpuMetrics;
      const timestamped = { ...result, timestamp: Date.now() };
      setData(timestamped);
      setHistory(prev => {
        // Keep last 60 data points (5 minutes at 5s interval)
        const updated = [...prev, timestamped];
        return updated.slice(-60);
      });
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to fetch GPU data');
    } finally {
      setLoading(false);
    }
  }, []);

  // Initial fetch
  useEffect(() => {
    fetchGpuData();
  }, [fetchGpuData]);

  // Polling effect
  useEffect(() => {
    if (isPolling && interval > 0) {
      intervalRef.current = setInterval(fetchGpuData, interval);
    }
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isPolling, interval, fetchGpuData]);

  const setPolling = useCallback((enabled: boolean) => {
    setIsPolling(enabled);
  }, []);

  const refresh = useCallback(async () => {
    setLoading(true);
    await fetchGpuData();
  }, [fetchGpuData]);

  return {
    data,
    history,
    loading,
    error,
    refresh,
    isPolling,
    setPolling,
  };
}

// Helper to calculate trends from history
export function calculateGpuTrends(history: GpuMetrics[]) {
  if (history.length < 2) return null;
  
  const recent = history.slice(-10);
  const avgTemp = recent.reduce((sum, m) => sum + m.temperature_gpu, 0) / recent.length;
  const avgUtil = recent.reduce((sum, m) => sum + m.utilization_gpu, 0) / recent.length;
  const avgPower = recent.reduce((sum, m) => sum + m.power_draw_w, 0) / recent.length;
  const avgMem = recent.reduce((sum, m) => sum + m.memory_used_gb, 0) / recent.length;
  
  const latest = history[history.length - 1];
  const oldest = history[0];
  
  return {
    avgTemperature: avgTemp,
    avgUtilization: avgUtil,
    avgPower: avgPower,
    avgMemoryUsed: avgMem,
    tempTrend: latest.temperature_gpu - oldest.temperature_gpu,
    utilTrend: latest.utilization_gpu - oldest.utilization_gpu,
    powerTrend: latest.power_draw_w - oldest.power_draw_w,
    memTrend: latest.memory_used_gb - oldest.memory_used_gb,
    dataPoints: history.length,
    timeSpan: latest.timestamp! - oldest.timestamp!,
  };
}




