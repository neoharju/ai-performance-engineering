'use client';

import { useState, useEffect, useCallback } from 'react';
import { 
  Lightbulb, 
  RefreshCw, 
  TrendingUp, 
  AlertTriangle, 
  CheckCircle, 
  XCircle,
  Trophy,
  HardDrive,
  Cpu,
  Zap,
  BookOpen,
  AlertCircle,
  Loader2,
  ChevronRight,
} from 'lucide-react';
import { getPerformanceInsights, refreshPerformanceInsights } from '@/lib/api';
import { cn } from '@/lib/utils';

interface Insight {
  id: string;
  type: 'success' | 'warning' | 'info' | 'error';
  icon: string;
  title: string;
  description: string;
  chapter?: string;
  count?: number;
  priority?: number;
}

const iconMap: Record<string, React.ReactNode> = {
  'trophy': <Trophy className="w-5 h-5" />,
  'trending-up': <TrendingUp className="w-5 h-5" />,
  'alert-triangle': <AlertTriangle className="w-5 h-5" />,
  'check-circle': <CheckCircle className="w-5 h-5" />,
  'x-circle': <XCircle className="w-5 h-5" />,
  'hard-drive': <HardDrive className="w-5 h-5" />,
  'cpu': <Cpu className="w-5 h-5" />,
  'zap': <Zap className="w-5 h-5" />,
  'book-open': <BookOpen className="w-5 h-5" />,
  'alert-circle': <AlertCircle className="w-5 h-5" />,
  'lightbulb': <Lightbulb className="w-5 h-5" />,
};

const typeColors: Record<string, { bg: string; border: string; text: string }> = {
  success: { bg: 'bg-accent-success/10', border: 'border-accent-success/30', text: 'text-accent-success' },
  warning: { bg: 'bg-accent-warning/10', border: 'border-accent-warning/30', text: 'text-accent-warning' },
  info: { bg: 'bg-accent-info/10', border: 'border-accent-info/30', text: 'text-accent-info' },
  error: { bg: 'bg-accent-danger/10', border: 'border-accent-danger/30', text: 'text-accent-danger' },
};

interface InsightsCardProps {
  className?: string;
  maxInsights?: number;
  compact?: boolean;
}

export function InsightsCard({ className, maxInsights = 6, compact = false }: InsightsCardProps) {
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [insights, setInsights] = useState<Insight[]>([]);
  const [cached, setCached] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadInsights = useCallback(async (forceRefresh = false) => {
    try {
      if (forceRefresh) {
        setRefreshing(true);
      } else {
        setLoading(true);
      }
      setError(null);
      
      const data = forceRefresh 
        ? await refreshPerformanceInsights()
        : await getPerformanceInsights();
      
      const result = data as any;
      setInsights(result.insights || []);
      setCached(result.cached || false);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load insights');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    loadInsights();
  }, [loadInsights]);

  const displayInsights = insights.slice(0, maxInsights);

  if (loading) {
    return (
      <div className={cn('card', className)}>
        <div className="card-body flex items-center justify-center py-12">
          <Loader2 className="w-6 h-6 animate-spin text-accent-primary" />
          <span className="ml-2 text-white/50">Analyzing performance...</span>
        </div>
      </div>
    );
  }

  if (compact) {
    return (
      <div className={cn('card', className)}>
        <div className="card-header">
          <div className="flex items-center gap-2">
            <Lightbulb className="w-5 h-5 text-accent-warning" />
            <h3 className="font-medium text-white">Insights</h3>
            {cached && <span className="text-xs text-white/30">(cached)</span>}
          </div>
          <button
            onClick={() => loadInsights(true)}
            disabled={refreshing}
            className="p-1.5 hover:bg-white/5 rounded-lg text-white/50 hover:text-white disabled:opacity-50"
          >
            <RefreshCw className={cn('w-4 h-4', refreshing && 'animate-spin')} />
          </button>
        </div>
        <div className="card-body space-y-2">
          {displayInsights.length === 0 ? (
            <p className="text-white/50 text-sm text-center py-4">No insights available</p>
          ) : (
            displayInsights.slice(0, 3).map((insight) => {
              const colors = typeColors[insight.type] || typeColors.info;
              return (
                <div
                  key={insight.id}
                  className={cn(
                    'flex items-center gap-3 p-2 rounded-lg border',
                    colors.bg,
                    colors.border
                  )}
                >
                  <span className={colors.text}>
                    {iconMap[insight.icon] || <Lightbulb className="w-4 h-4" />}
                  </span>
                  <span className="text-sm text-white truncate flex-1">{insight.title}</span>
                </div>
              );
            })
          )}
        </div>
      </div>
    );
  }

  return (
    <div className={cn('card', className)}>
      <div className="card-header">
        <div className="flex items-center gap-2">
          <Lightbulb className="w-5 h-5 text-accent-warning" />
          <h2 className="text-lg font-semibold text-white">Performance Insights</h2>
          {cached && <span className="text-xs text-white/30">(cached)</span>}
        </div>
        <div className="flex items-center gap-2">
          <span className="text-sm text-white/50">{insights.length} insights</span>
          <button
            onClick={() => loadInsights(true)}
            disabled={refreshing}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-accent-primary/20 text-accent-primary rounded-lg text-sm hover:bg-accent-primary/30 disabled:opacity-50"
          >
            <RefreshCw className={cn('w-3 h-3', refreshing && 'animate-spin')} />
            Refresh
          </button>
        </div>
      </div>

      {error && (
        <div className="px-5 py-3 bg-accent-danger/10 border-t border-accent-danger/20 text-sm text-accent-danger">
          {error}
        </div>
      )}

      <div className="card-body">
        {displayInsights.length === 0 ? (
          <div className="text-center py-8">
            <Lightbulb className="w-12 h-12 mx-auto mb-4 text-white/20" />
            <p className="text-white/50">No insights available</p>
            <p className="text-sm text-white/30 mt-1">Run some benchmarks to generate insights</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {displayInsights.map((insight) => {
              const colors = typeColors[insight.type] || typeColors.info;
              return (
                <div
                  key={insight.id}
                  className={cn(
                    'p-4 rounded-xl border transition-all hover:scale-[1.02]',
                    colors.bg,
                    colors.border
                  )}
                >
                  <div className="flex items-start gap-3">
                    <div className={cn('p-2 rounded-lg', colors.bg, colors.text)}>
                      {iconMap[insight.icon] || <Lightbulb className="w-5 h-5" />}
                    </div>
                    <div className="flex-1 min-w-0">
                      <h3 className="font-medium text-white mb-1">{insight.title}</h3>
                      <p className="text-sm text-white/60">{insight.description}</p>
                      {insight.chapter && (
                        <div className="flex items-center gap-1 mt-2 text-xs text-white/40">
                          <BookOpen className="w-3 h-3" />
                          {insight.chapter}
                        </div>
                      )}
                    </div>
                    <ChevronRight className="w-4 h-4 text-white/20" />
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}




