'use client';

import { useState, useEffect, useCallback } from 'react';
import { getCostSavingsHeader } from '@/lib/api';
import { cn } from '@/lib/utils';
import {
  DollarSign,
  TrendingUp,
  ChevronDown,
  Zap,
  Settings,
  RefreshCw,
  X,
} from 'lucide-react';

interface SavingsData {
  total_monthly_savings_usd: number;
  total_yearly_savings_usd: number;
  avg_speedup: number;
  avg_time_saved_pct: number;
  successful_optimizations: number;
  gpu: {
    name: string;
    type: string;
    hourly_rate_usd: number;
  };
  assumptions: {
    ops_per_day: number;
    ops_per_month: number;
    cloud_provider: string;
    pricing_source: string;
  };
  top_savers: Array<{
    name: string;
    speedup: number;
    time_saved_pct: number;
    monthly_savings_usd: number;
    yearly_savings_usd: number;
  }>;
  pricing_table: Record<string, number>;
}

function formatCurrency(amount: number, compact: boolean = false): string {
  if (compact) {
    if (amount >= 1_000_000) {
      return `$${(amount / 1_000_000).toFixed(1)}M`;
    } else if (amount >= 1_000) {
      return `$${(amount / 1_000).toFixed(1)}k`;
    }
    return `$${amount.toFixed(0)}`;
  }
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(amount);
}

function formatNumber(num: number): string {
  if (num >= 1_000_000_000) {
    return `${(num / 1_000_000_000).toFixed(1)}B`;
  } else if (num >= 1_000_000) {
    return `${(num / 1_000_000).toFixed(1)}M`;
  } else if (num >= 1_000) {
    return `${(num / 1_000).toFixed(1)}K`;
  }
  return num.toString();
}

export function SavingsHeaderWidget() {
  const [savings, setSavings] = useState<SavingsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showDropdown, setShowDropdown] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [opsPerDay, setOpsPerDay] = useState(1_000_000);
  const [displayMode, setDisplayMode] = useState<'monthly' | 'yearly'>('monthly');

  const loadSavings = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getCostSavingsHeader(opsPerDay) as SavingsData;
      setSavings(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load savings');
    } finally {
      setLoading(false);
    }
  }, [opsPerDay]);

  useEffect(() => {
    loadSavings();
    // Refresh every 60 seconds
    const interval = setInterval(loadSavings, 60000);
    return () => clearInterval(interval);
  }, [loadSavings]);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (!target.closest('.savings-widget')) {
        setShowDropdown(false);
        setShowSettings(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  if (loading && !savings) {
    return (
      <div className="savings-widget flex items-center gap-2 px-3 py-1.5 rounded-xl bg-white/5 border border-white/10 animate-pulse">
        <DollarSign className="w-4 h-4 text-white/40" />
        <span className="text-sm text-white/40">Loading...</span>
      </div>
    );
  }

  if (error || !savings) {
    return (
      <button
        onClick={loadSavings}
        className="savings-widget flex items-center gap-2 px-3 py-1.5 rounded-xl bg-white/5 border border-white/10 text-white/60 hover:text-white hover:bg-white/10 transition-all"
        title="Failed to load savings - click to retry"
      >
        <DollarSign className="w-4 h-4" />
        <span className="text-sm">--</span>
      </button>
    );
  }

  const displayValue = displayMode === 'monthly' 
    ? savings.total_monthly_savings_usd 
    : savings.total_yearly_savings_usd;
  
  const hasSavings = displayValue > 0;

  return (
    <div className="savings-widget relative">
      {/* Main Button */}
      <button
        onClick={() => setShowDropdown(!showDropdown)}
        className={cn(
          'flex items-center gap-2 px-3 py-1.5 rounded-xl transition-all',
          hasSavings
            ? 'bg-gradient-to-r from-emerald-500/20 to-green-500/20 border border-emerald-500/40 hover:border-emerald-500/60 shadow-lg shadow-emerald-500/10'
            : 'bg-white/5 border border-white/10 hover:bg-white/10'
        )}
      >
        <div className={cn(
          'w-6 h-6 rounded-lg flex items-center justify-center',
          hasSavings ? 'bg-emerald-500/30' : 'bg-white/10'
        )}>
          <DollarSign className={cn(
            'w-4 h-4',
            hasSavings ? 'text-emerald-400' : 'text-white/60'
          )} />
        </div>
        
        <div className="flex flex-col items-start">
          <span className={cn(
            'text-base font-bold leading-tight',
            hasSavings ? 'text-emerald-400' : 'text-white/60'
          )}>
            {formatCurrency(displayValue, true)}
            <span className="text-xs font-normal text-white/50">
              /{displayMode === 'monthly' ? 'mo' : 'yr'}
            </span>
          </span>
          <span className="text-[10px] text-white/40 leading-tight">
            savings
          </span>
        </div>

        <ChevronDown className={cn(
          'w-3 h-3 text-white/40 transition-transform',
          showDropdown && 'rotate-180'
        )} />
      </button>

      {/* Dropdown Panel */}
      {showDropdown && (
        <div className="absolute top-full right-0 mt-2 w-[380px] bg-brand-card border border-white/10 rounded-2xl shadow-2xl z-[9999] overflow-hidden animate-slide-in">
          {/* Header */}
          <div className="p-4 bg-gradient-to-r from-emerald-500/10 to-green-500/10 border-b border-white/5">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-white font-semibold flex items-center gap-2">
                <DollarSign className="w-5 h-5 text-emerald-400" />
                Total Savings
              </h3>
              <div className="flex items-center gap-1">
                <button
                  onClick={() => setShowSettings(!showSettings)}
                  className="p-1.5 rounded-lg hover:bg-white/10 transition-colors"
                  title="Settings"
                >
                  <Settings className="w-4 h-4 text-white/60" />
                </button>
                <button
                  onClick={loadSavings}
                  className="p-1.5 rounded-lg hover:bg-white/10 transition-colors"
                  title="Refresh"
                >
                  <RefreshCw className={cn(
                    'w-4 h-4 text-white/60',
                    loading && 'animate-spin'
                  )} />
                </button>
                <button
                  onClick={() => setShowDropdown(false)}
                  className="p-1.5 rounded-lg hover:bg-white/10 transition-colors"
                >
                  <X className="w-4 h-4 text-white/60" />
                </button>
              </div>
            </div>

            {/* Big Numbers */}
            <div className="grid grid-cols-2 gap-4">
              <button
                onClick={() => setDisplayMode('monthly')}
                className={cn(
                  'p-3 rounded-xl transition-all text-left',
                  displayMode === 'monthly'
                    ? 'bg-emerald-500/20 border-2 border-emerald-500/50'
                    : 'bg-white/5 border-2 border-transparent hover:border-white/20'
                )}
              >
                <div className="text-xs text-white/60 mb-1">Monthly</div>
                <div className={cn(
                  'text-2xl font-bold',
                  displayMode === 'monthly' ? 'text-emerald-400' : 'text-white'
                )}>
                  {formatCurrency(savings.total_monthly_savings_usd)}
                </div>
              </button>
              <button
                onClick={() => setDisplayMode('yearly')}
                className={cn(
                  'p-3 rounded-xl transition-all text-left',
                  displayMode === 'yearly'
                    ? 'bg-emerald-500/20 border-2 border-emerald-500/50'
                    : 'bg-white/5 border-2 border-transparent hover:border-white/20'
                )}
              >
                <div className="text-xs text-white/60 mb-1">Yearly</div>
                <div className={cn(
                  'text-2xl font-bold',
                  displayMode === 'yearly' ? 'text-emerald-400' : 'text-white'
                )}>
                  {formatCurrency(savings.total_yearly_savings_usd)}
                </div>
              </button>
            </div>
          </div>

          {/* Settings Panel */}
          {showSettings && (
            <div className="p-4 bg-white/5 border-b border-white/5 space-y-3">
              <div className="flex items-center justify-between">
                <label className="text-sm text-white/70">Operations per day</label>
                <div className="flex items-center gap-2">
                  <input
                    type="number"
                    value={opsPerDay}
                    onChange={(e) => setOpsPerDay(Number(e.target.value) || 1_000_000)}
                    className="w-28 px-2 py-1 bg-white/10 border border-white/20 rounded text-white text-sm text-right"
                    min={1000}
                    step={100000}
                  />
                  <button
                    onClick={loadSavings}
                    className="px-2 py-1 bg-emerald-500/20 text-emerald-400 rounded text-sm hover:bg-emerald-500/30"
                  >
                    Apply
                  </button>
                </div>
              </div>
              <div className="text-xs text-white/50">
                Presets: 
                <button onClick={() => { setOpsPerDay(100_000); }} className="ml-2 px-2 py-0.5 rounded bg-white/10 hover:bg-white/20">100K</button>
                <button onClick={() => { setOpsPerDay(1_000_000); }} className="ml-1 px-2 py-0.5 rounded bg-white/10 hover:bg-white/20">1M</button>
                <button onClick={() => { setOpsPerDay(10_000_000); }} className="ml-1 px-2 py-0.5 rounded bg-white/10 hover:bg-white/20">10M</button>
                <button onClick={() => { setOpsPerDay(100_000_000); }} className="ml-1 px-2 py-0.5 rounded bg-white/10 hover:bg-white/20">100M</button>
              </div>
            </div>
          )}

          {/* Stats Row */}
          <div className="grid grid-cols-3 gap-px bg-white/5">
            <div className="p-3 bg-brand-card">
              <div className="text-xs text-white/50 mb-1">Avg Speedup</div>
              <div className="text-lg font-bold text-accent-primary flex items-center gap-1">
                <Zap className="w-4 h-4" />
                {savings.avg_speedup.toFixed(1)}x
              </div>
            </div>
            <div className="p-3 bg-brand-card">
              <div className="text-xs text-white/50 mb-1">Time Saved</div>
              <div className="text-lg font-bold text-accent-success flex items-center gap-1">
                <TrendingUp className="w-4 h-4" />
                {savings.avg_time_saved_pct.toFixed(0)}%
              </div>
            </div>
            <div className="p-3 bg-brand-card">
              <div className="text-xs text-white/50 mb-1">Optimizations</div>
              <div className="text-lg font-bold text-white">
                {savings.successful_optimizations}
              </div>
            </div>
          </div>

          {/* GPU Info */}
          <div className="p-3 border-b border-white/5 bg-white/5">
            <div className="flex items-center justify-between text-sm">
              <span className="text-white/60">GPU</span>
              <span className="text-white font-medium">{savings.gpu.type}</span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-white/60">Hourly Rate</span>
              <span className="text-emerald-400 font-medium">${savings.gpu.hourly_rate_usd.toFixed(2)}/hr</span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-white/60">Scale</span>
              <span className="text-white font-medium">{formatNumber(savings.assumptions.ops_per_day)} ops/day</span>
            </div>
          </div>

          {/* Top Savers */}
          {savings.top_savers.length > 0 && (
            <div className="p-3">
              <div className="text-xs text-white/50 uppercase tracking-wider mb-2">
                Top Savers
              </div>
              <div className="space-y-2 max-h-40 overflow-y-auto">
                {savings.top_savers.map((saver, idx) => (
                  <div
                    key={idx}
                    className="flex items-center justify-between py-1.5 px-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
                  >
                    <div className="flex-1 min-w-0">
                      <div className="text-sm text-white truncate">{saver.name.split(':')[1]}</div>
                      <div className="text-xs text-white/50">{saver.speedup}x speedup</div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm font-medium text-emerald-400">
                        {formatCurrency(saver.monthly_savings_usd)}/mo
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Footer */}
          <div className="px-3 py-2 bg-white/5 border-t border-white/5 text-[10px] text-white/40">
            Based on {savings.assumptions.pricing_source} • {savings.assumptions.cloud_provider}
          </div>
        </div>
      )}
    </div>
  );
}




