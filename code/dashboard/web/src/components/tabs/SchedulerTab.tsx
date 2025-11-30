'use client';

import { useState, useEffect, useCallback } from 'react';
import { 
  Calendar, 
  Clock, 
  Plus, 
  Trash2, 
  Play, 
  Pause, 
  RefreshCw,
  Loader2,
  AlertTriangle,
  CheckCircle,
  History,
  Bell,
  Target,
} from 'lucide-react';
import { 
  getScheduledJobs, 
  getSchedulerHistory, 
  createScheduledJob, 
  deleteScheduledJob,
  toggleScheduledJob,
  runScheduledJobNow,
  getBenchmarkData,
} from '@/lib/api';
import { cn } from '@/lib/utils';
import { useToast } from '@/components/Toast';

interface ScheduledJob {
  id: string;
  name: string;
  targets: string[];
  schedule_type: 'once' | 'daily' | 'weekly' | 'interval';
  schedule_time: string;
  schedule_days: number[];
  interval_minutes: number;
  enabled: boolean;
  created_at: string;
  last_run: string | null;
  last_status: string | null;
  next_run: string;
  notify_on_complete: boolean;
  notify_on_failure: boolean;
}

interface HistoryEntry {
  job_id: string;
  job_name: string;
  ran_at: string;
  status: string;
  duration_seconds: number;
}

const dayNames = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

export function SchedulerTab() {
  const [loading, setLoading] = useState(true);
  const [jobs, setJobs] = useState<ScheduledJob[]>([]);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [availableTargets, setAvailableTargets] = useState<string[]>([]);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [creating, setCreating] = useState(false);
  const { showToast } = useToast();

  // New job form state
  const [newJob, setNewJob] = useState({
    name: '',
    targets: [] as string[],
    schedule_type: 'daily' as 'once' | 'daily' | 'weekly' | 'interval',
    schedule_time: '02:00',
    schedule_days: [] as number[],
    interval_minutes: 60,
    notify_on_complete: false,
    notify_on_failure: true,
  });

  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      const [jobsData, historyData, benchmarkData] = await Promise.all([
        getScheduledJobs(),
        getSchedulerHistory(),
        getBenchmarkData(),
      ]);

      setJobs((jobsData as any).jobs || []);
      setHistory((historyData as any).history || []);

      // Extract unique chapters as targets
      const benchmarks = (benchmarkData as any).benchmarks || [];
      const chapters = [...new Set(benchmarks.map((b: any) => b.chapter))].filter(Boolean);
      setAvailableTargets(chapters as string[]);
    } catch (e) {
      showToast('Failed to load scheduler data', 'error');
    } finally {
      setLoading(false);
    }
  }, [showToast]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleCreate = async () => {
    if (!newJob.name || newJob.targets.length === 0) {
      showToast('Please provide a name and select targets', 'error');
      return;
    }

    try {
      setCreating(true);
      await createScheduledJob(newJob);
      showToast('Scheduled job created', 'success');
      setShowCreateForm(false);
      setNewJob({
        name: '',
        targets: [],
        schedule_type: 'daily',
        schedule_time: '02:00',
        schedule_days: [],
        interval_minutes: 60,
        notify_on_complete: false,
        notify_on_failure: true,
      });
      loadData();
    } catch (e) {
      showToast('Failed to create job', 'error');
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await deleteScheduledJob(id);
      showToast('Job deleted', 'success');
      loadData();
    } catch (e) {
      showToast('Failed to delete job', 'error');
    }
  };

  const handleToggle = async (id: string) => {
    try {
      await toggleScheduledJob(id);
      loadData();
    } catch (e) {
      showToast('Failed to toggle job', 'error');
    }
  };

  const handleRunNow = async (id: string) => {
    try {
      await runScheduledJobNow(id);
      showToast('Job started', 'success');
      loadData();
    } catch (e) {
      showToast('Failed to run job', 'error');
    }
  };

  const formatNextRun = (isoString: string) => {
    try {
      const date = new Date(isoString);
      return date.toLocaleString();
    } catch {
      return 'Unknown';
    }
  };

  if (loading) {
    return (
      <div className="card">
        <div className="card-body flex items-center justify-center py-20">
          <Loader2 className="w-8 h-8 animate-spin text-accent-primary" />
          <span className="ml-3 text-white/50">Loading scheduler...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center gap-2">
            <Calendar className="w-5 h-5 text-accent-primary" />
            <h2 className="text-lg font-semibold text-white">Benchmark Scheduler</h2>
          </div>
          <div className="flex items-center gap-2">
            <button onClick={loadData} className="p-2 hover:bg-white/5 rounded-lg">
              <RefreshCw className="w-4 h-4 text-white/50" />
            </button>
            <button
              onClick={() => setShowCreateForm(true)}
              className="flex items-center gap-2 px-4 py-2 bg-accent-primary/20 text-accent-primary rounded-lg hover:bg-accent-primary/30"
            >
              <Plus className="w-4 h-4" />
              Schedule Benchmark
            </button>
          </div>
        </div>
      </div>

      {/* Create Form */}
      {showCreateForm && (
        <div className="card">
          <div className="card-header">
            <h3 className="font-medium text-white">New Scheduled Job</h3>
          </div>
          <div className="card-body space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm text-white/50 mb-2">Job Name</label>
                <input
                  type="text"
                  value={newJob.name}
                  onChange={(e) => setNewJob({ ...newJob, name: e.target.value })}
                  placeholder="e.g., Nightly Regression Test"
                  className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder:text-white/30 focus:outline-none focus:border-accent-primary/50"
                />
              </div>
              <div>
                <label className="block text-sm text-white/50 mb-2">Schedule Type</label>
                <select
                  value={newJob.schedule_type}
                  onChange={(e) => setNewJob({ ...newJob, schedule_type: e.target.value as any })}
                  className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:border-accent-primary/50"
                >
                  <option value="once" className="bg-brand-bg">Run Once</option>
                  <option value="daily" className="bg-brand-bg">Daily</option>
                  <option value="weekly" className="bg-brand-bg">Weekly</option>
                  <option value="interval" className="bg-brand-bg">Interval</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {newJob.schedule_type !== 'interval' && (
                <div>
                  <label className="block text-sm text-white/50 mb-2">Time</label>
                  <input
                    type="time"
                    value={newJob.schedule_time}
                    onChange={(e) => setNewJob({ ...newJob, schedule_time: e.target.value })}
                    className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:border-accent-primary/50"
                  />
                </div>
              )}
              {newJob.schedule_type === 'interval' && (
                <div>
                  <label className="block text-sm text-white/50 mb-2">Interval (minutes)</label>
                  <input
                    type="number"
                    value={newJob.interval_minutes}
                    onChange={(e) => setNewJob({ ...newJob, interval_minutes: parseInt(e.target.value) || 60 })}
                    min={5}
                    className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:border-accent-primary/50"
                  />
                </div>
              )}
            </div>

            {newJob.schedule_type === 'weekly' && (
              <div>
                <label className="block text-sm text-white/50 mb-2">Days</label>
                <div className="flex gap-2">
                  {dayNames.map((day, i) => (
                    <button
                      key={day}
                      onClick={() => {
                        const days = newJob.schedule_days.includes(i)
                          ? newJob.schedule_days.filter(d => d !== i)
                          : [...newJob.schedule_days, i];
                        setNewJob({ ...newJob, schedule_days: days });
                      }}
                      className={cn(
                        'px-3 py-2 rounded-lg text-sm transition-colors',
                        newJob.schedule_days.includes(i)
                          ? 'bg-accent-primary/20 text-accent-primary border border-accent-primary/30'
                          : 'bg-white/5 text-white/60 hover:bg-white/10'
                      )}
                    >
                      {day}
                    </button>
                  ))}
                </div>
              </div>
            )}

            <div>
              <label className="block text-sm text-white/50 mb-2">Targets (Chapters)</label>
              <div className="flex flex-wrap gap-2">
                {availableTargets.map((target) => (
                  <button
                    key={target}
                    onClick={() => {
                      const targets = newJob.targets.includes(target)
                        ? newJob.targets.filter(t => t !== target)
                        : [...newJob.targets, target];
                      setNewJob({ ...newJob, targets });
                    }}
                    className={cn(
                      'px-3 py-2 rounded-lg text-sm transition-colors',
                      newJob.targets.includes(target)
                        ? 'bg-accent-success/20 text-accent-success border border-accent-success/30'
                        : 'bg-white/5 text-white/60 hover:bg-white/10'
                    )}
                  >
                    <Target className="w-3 h-3 inline mr-1" />
                    {target}
                  </button>
                ))}
              </div>
            </div>

            <div className="flex items-center gap-6">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={newJob.notify_on_complete}
                  onChange={(e) => setNewJob({ ...newJob, notify_on_complete: e.target.checked })}
                  className="w-4 h-4 accent-accent-primary"
                />
                <span className="text-sm text-white/70">Notify on complete</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={newJob.notify_on_failure}
                  onChange={(e) => setNewJob({ ...newJob, notify_on_failure: e.target.checked })}
                  className="w-4 h-4 accent-accent-primary"
                />
                <span className="text-sm text-white/70">Notify on failure</span>
              </label>
            </div>

            <div className="flex gap-2 pt-4 border-t border-white/5">
              <button
                onClick={handleCreate}
                disabled={creating}
                className="px-4 py-2 bg-accent-primary text-black rounded-lg font-medium disabled:opacity-50 hover:opacity-90"
              >
                {creating ? 'Creating...' : 'Create Job'}
              </button>
              <button
                onClick={() => setShowCreateForm(false)}
                className="px-4 py-2 bg-white/5 text-white rounded-lg hover:bg-white/10"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Jobs List */}
      <div className="card">
        <div className="card-header">
          <h3 className="font-medium text-white">Scheduled Jobs</h3>
          <span className="text-sm text-white/50">{jobs.length} jobs</span>
        </div>
        <div className="card-body">
          {jobs.length === 0 ? (
            <div className="text-center py-12">
              <Calendar className="w-12 h-12 mx-auto mb-4 text-white/20" />
              <p className="text-white/50">No scheduled jobs</p>
              <p className="text-sm text-white/30 mt-1">Create a scheduled job to automate benchmark runs</p>
            </div>
          ) : (
            <div className="space-y-3">
              {jobs.map((job) => (
                <div
                  key={job.id}
                  className={cn(
                    'p-4 rounded-xl border transition-all',
                    job.enabled
                      ? 'bg-white/5 border-white/10'
                      : 'bg-white/[0.02] border-white/5 opacity-60'
                  )}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        <h4 className="font-medium text-white">{job.name}</h4>
                        <span className={cn(
                          'px-2 py-0.5 rounded text-xs',
                          job.enabled
                            ? 'bg-accent-success/20 text-accent-success'
                            : 'bg-white/10 text-white/50'
                        )}>
                          {job.enabled ? 'Active' : 'Paused'}
                        </span>
                      </div>
                      <div className="flex items-center gap-4 text-sm text-white/50">
                        <span className="flex items-center gap-1">
                          <Clock className="w-3 h-3" />
                          {job.schedule_type === 'interval'
                            ? `Every ${job.interval_minutes} min`
                            : `${job.schedule_type} at ${job.schedule_time}`}
                        </span>
                        <span className="flex items-center gap-1">
                          <Target className="w-3 h-3" />
                          {job.targets.length} target{job.targets.length !== 1 ? 's' : ''}
                        </span>
                        {job.notify_on_failure && (
                          <span className="flex items-center gap-1">
                            <Bell className="w-3 h-3" />
                            Alerts on
                          </span>
                        )}
                      </div>
                      <div className="mt-2 text-xs text-white/40">
                        Next run: {formatNextRun(job.next_run)}
                        {job.last_run && (
                          <span className="ml-3">
                            Last run: {formatNextRun(job.last_run)}
                            {job.last_status && (
                              <span className={cn(
                                'ml-1',
                                job.last_status === 'completed' ? 'text-accent-success' : 'text-accent-danger'
                              )}>
                                ({job.last_status})
                              </span>
                            )}
                          </span>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => handleRunNow(job.id)}
                        className="p-2 hover:bg-accent-success/10 rounded-lg text-accent-success"
                        title="Run now"
                      >
                        <Play className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => handleToggle(job.id)}
                        className="p-2 hover:bg-white/5 rounded-lg text-white/50"
                        title={job.enabled ? 'Pause' : 'Resume'}
                      >
                        {job.enabled ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                      </button>
                      <button
                        onClick={() => handleDelete(job.id)}
                        className="p-2 hover:bg-accent-danger/10 rounded-lg text-accent-danger"
                        title="Delete"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* History */}
      {history.length > 0 && (
        <div className="card">
          <div className="card-header">
            <div className="flex items-center gap-2">
              <History className="w-5 h-5 text-accent-info" />
              <h3 className="font-medium text-white">Run History</h3>
            </div>
          </div>
          <div className="card-body">
            <div className="space-y-2">
              {history.slice(0, 10).map((entry, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between p-3 rounded-lg bg-white/5"
                >
                  <div className="flex items-center gap-3">
                    {entry.status === 'completed' ? (
                      <CheckCircle className="w-4 h-4 text-accent-success" />
                    ) : (
                      <AlertTriangle className="w-4 h-4 text-accent-danger" />
                    )}
                    <span className="text-white">{entry.job_name}</span>
                  </div>
                  <div className="text-sm text-white/50">
                    {formatNextRun(entry.ran_at)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}




