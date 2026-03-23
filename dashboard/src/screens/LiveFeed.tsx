import { motion, AnimatePresence } from 'framer-motion';
import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Activity, Clock, ArrowUpRight, GitBranch, ExternalLink, AlertTriangle, Sparkles, Server } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import { useRuns } from '../hooks/usePipelineData';
import { ErrorTypeBadge } from '../components/shared/ErrorTypeBadge';
import { Badge } from '../components/ui/Badge';
import { StatusDot } from '../components/ui/StatusDot';
import { ProgressBar } from '../components/ui/ProgressBar';
import { NODE_ORDER, NODE_LABELS } from '../lib/constants';
import type { PipelineRun, RunStatus } from '../types/pipeline';
import type { Outcome } from '../lib/api';

/* ═══════════════════════════════════════════════════════════════════════════
   STATUS DOT COLOR MAP
   ═══════════════════════════════════════════════════════════════════════════ */
function statusDotColor(status: RunStatus): string {
    switch (status) {
        case 'ingesting': case 'extracting': return '#3B82F6';
        case 'classifying': case 'analyzing': return '#8B5CF6';
        case 'fixing': case 'remediating': return '#F59E0B';
        case 'paused_review': return '#F43F5E';
        case 'completed': return '#10B981';
        case 'failed': return '#F43F5E';
        default: return '#6366F1';
    }
}

const OUTCOME_COLORS: Record<Outcome, { bg: string; text: string }> = {
    'Auto-fixed': { bg: 'rgba(16,185,129,0.15)', text: '#34D399' },
    'Re-run triggered': { bg: 'rgba(59,130,246,0.15)', text: '#60A5FA' },
    'PR opened': { bg: 'rgba(99,102,241,0.15)', text: '#818CF8' },
    'Human reviewed': { bg: 'rgba(139,92,246,0.15)', text: '#A78BFA' },
    'Escalated': { bg: 'rgba(244,63,94,0.15)', text: '#FB7185' },
    'No action': { bg: 'rgba(107,107,128,0.12)', text: '#9CA3AF' },
};

/* ═══════════════════════════════════════════════════════════════════════════
   NODE DOT PIPELINE
   ═══════════════════════════════════════════════════════════════════════════ */
function NodeDots({ currentNode }: { currentNode?: string }) {
    const currentIdx = currentNode ? NODE_ORDER.indexOf(currentNode) : -1;
    return (
        <div className="flex items-center gap-[5px]" aria-label="Node progress dots">
            {NODE_ORDER.map((node, idx) => {
                const isDone = currentIdx >= 0 && idx < currentIdx;
                const isCurrent = idx === currentIdx;
                const dotColor = isDone ? '#6366F1' : isCurrent ? '#6366F1' : '#22222F';
                return (
                    <div key={node} className="relative group/dot">
                        <div
                            className={`w-2 h-2 rounded-full transition-all duration-300 ${isCurrent ? 'animate-pulse' : ''}`}
                            style={{ background: dotColor, boxShadow: isCurrent ? '0 0 6px rgba(99,102,241,0.6)' : 'none' }}
                        />
                        <div
                            className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1.5 px-2 py-1 rounded text-xs whitespace-nowrap opacity-0 group-hover/dot:opacity-100 pointer-events-none transition-opacity duration-150 z-10"
                            style={{ background: '#1A1A24', border: '1px solid rgba(255,255,255,0.1)', color: '#A1A1B5' }}
                        >
                            {NODE_LABELS[node] ?? node}
                        </div>
                    </div>
                );
            })}
        </div>
    );
}

/* ═══════════════════════════════════════════════════════════════════════════
   ACTIVE RUN CARD
   ═══════════════════════════════════════════════════════════════════════════ */
function RunCard({ run, index }: { run: PipelineRun; index: number }) {
    const navigate = useNavigate();
    const isPaused = run.status === 'paused_review';
    const elapsed = Math.round((Date.now() - new Date(run.startedAt).getTime()) / 1000);
    const minutes = Math.max(0, Math.floor(elapsed / 60));
    const seconds = Math.max(0, elapsed % 60);
    const dotColor = statusDotColor(run.status);

    return (
        <motion.div
            layout
            initial={{ opacity: 0, y: -20, scale: 0.98 }}
            animate={{
                opacity: 1, y: 0, scale: 1,
                borderColor: isPaused ? ['rgba(244,63,94,0.15)', 'rgba(244,63,94,0.4)', 'rgba(244,63,94,0.15)'] : 'rgba(255,255,255,0.08)',
            }}
            transition={{
                delay: index * 0.08,
                duration: 0.35,
                ease: [0.16, 1, 0.3, 1],
                borderColor: isPaused ? { repeat: Infinity, duration: 2.5 } : undefined,
            }}
            className="glass-card p-4 relative overflow-hidden"
            style={{ borderWidth: 1 }}
        >
            <div className="flex items-center gap-3 mb-3">
                <span className="animate-pulse" style={{ width: 8, height: 8, borderRadius: '50%', background: dotColor, boxShadow: `0 0 6px ${dotColor}`, flexShrink: 0 }} />
                <span className="font-medium text-white text-sm">{run.repo}</span>
                {run.errorType && <ErrorTypeBadge type={run.errorType} />}
                <span className="text-xs text-label flex items-center gap-1 ml-auto">
                    <GitBranch size={10} />{run.branch} · {run.stage}
                </span>
            </div>

            <div className="flex items-center gap-3 mb-2.5">
                <div className="flex-1">
                    <ProgressBar value={run.nodeProgress ?? 0} color="#6366F1" height={3} shimmer className="mb-2" />
                    <div className="flex items-center justify-between">
                        <NodeDots currentNode={run.currentNode} />
                        <span className="text-xs text-label font-mono ml-2 shrink-0">
                            {NODE_LABELS[run.currentNode ?? ''] ?? run.currentNode ?? '…'} · {run.nodeProgress ?? 0}%
                        </span>
                    </div>
                </div>
            </div>

            <div className="flex items-center justify-between mt-4 border-t border-white/5 pt-3">
                <span className="flex items-center gap-1.5 text-xs text-label">
                    <Sparkles size={10} style={{ color: '#8B5CF6' }} />
                    <span>Gemini Agent</span>
                    <span className="animate-pulse" style={{ color: '#8B5CF6' }}>· actively tracing...</span>
                </span>
                <div className="flex items-center gap-3">
                    <span className="text-xs text-label font-mono">
                        {minutes}:{seconds.toString().padStart(2, '0')}
                    </span>
                    <button onClick={() => navigate(`/run/${run.id}`)} className="flex items-center gap-1 text-xs font-medium transition-all duration-200 hover:text-white px-2 py-1 rounded-md hover:bg-white/5" style={{ color: '#818CF8' }}>
                        <ExternalLink size={11} /> view
                    </button>
                </div>
            </div>
        </motion.div>
    );
}

/* ═══════════════════════════════════════════════════════════════════════════
   COMPLETED RUNS TABLE
   ═══════════════════════════════════════════════════════════════════════════ */
function CompletedTable({ runs }: { runs: (PipelineRun & { outcome?: Outcome })[] }) {
    const navigate = useNavigate();
    const [visibleCount, setVisibleCount] = useState(20);
    const visible = runs.slice(0, visibleCount);

    if (runs.length === 0) return null;

    return (
        <div className="mt-8">
            <h3 className="label-text mb-4 px-1">Recent Completed</h3>
            <div className="grid grid-cols-[20px_1fr_auto_80px_auto_70px_50px] gap-3 items-center px-4 py-2 text-xs text-label mb-1">
                <span /><span>Repo</span><span>Error Type</span><span>Severity</span><span>Outcome</span><span>Time</span><span />
            </div>

            <div role="table">
                <AnimatePresence>
                    {visible.map((run, i) => {
                        const outcome = run.outcome ?? 'No action';
                        const oColors = OUTCOME_COLORS[outcome];
                        return (
                            <motion.div key={run.id} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 16 }} transition={{ delay: i * 0.025 }} onClick={() => navigate(`/run/${run.id}`)} className="grid grid-cols-[20px_1fr_auto_80px_auto_70px_50px] gap-3 items-center px-4 py-3 rounded-lg cursor-pointer transition-all duration-200 hover:bg-[rgba(99,102,241,0.05)]" style={{ background: i % 2 === 1 ? 'rgba(255,255,255,0.015)' : 'transparent' }}>
                                <StatusDot severity={run.severity} />
                                <span className="text-sm text-white font-medium truncate">{run.repo}</span>
                                <span>{run.errorType && <ErrorTypeBadge type={run.errorType} />}</span>
                                <span>{run.severity && <Badge label={run.severity.toUpperCase()} severity={run.severity} />}</span>
                                <span className="inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium" style={{ background: oColors.bg, color: oColors.text }}>{outcome}</span>
                                <span className="text-xs text-label font-mono">{formatDistanceToNow(new Date(run.completedAt ?? run.startedAt), { addSuffix: false })}</span>
                                <span className="flex justify-end"><ArrowUpRight size={12} className="text-label" /></span>
                            </motion.div>
                        );
                    })}
                </AnimatePresence>
            </div>

            {visibleCount < runs.length && (
                <div className="flex justify-center mt-4">
                    <button onClick={() => setVisibleCount(c => c + 20)} className="btn-secondary text-xs">Load more</button>
                </div>
            )}
        </div>
    );
}

/* ═══════════════════════════════════════════════════════════════════════════
   LIVE FEED SCREEN
   ═══════════════════════════════════════════════════════════════════════════ */
export function LiveFeed() {
    const { data: allRuns = [] } = useRuns();

    const activeRuns = useMemo(() => allRuns.filter(r => !['completed', 'failed'].includes(r.status)), [allRuns]);
    const completedRuns = useMemo(() => (allRuns.filter(r => ['completed', 'failed'].includes(r.status)) as (PipelineRun & { outcome?: Outcome })[]).sort((a, b) => new Date(b.completedAt ?? 0).getTime() - new Date(a.completedAt ?? 0).getTime()), [allRuns]);

    return (
        <div className="max-w-5xl mx-auto animate-fade-in relative z-0">
            <div className="mb-8 pl-1">
                <h1 className="text-2xl font-bold text-white tracking-tight flex items-center gap-2">
                    <Activity className="text-indigo-400" /> Live Pipeline Feed
                </h1>
                <p className="text-sm text-slate-400 mt-1">Listening for CI/CD failures via Google Cloud Platform</p>
            </div>

            {allRuns.length === 0 ? (
                <div className="mt-16 flex flex-col items-center justify-center py-20 px-4 text-center bg-white/5 border border-dashed border-slate-700/50 rounded-2xl shadow-sm relative overflow-hidden">
                    <div className="absolute inset-0 bg-indigo-500/5 blur-[100px]" />
                    <div className="w-16 h-16 rounded-full bg-slate-800/50 flex items-center justify-center mb-6 relative border border-slate-700">
                        <Server className="w-8 h-8 text-indigo-400 animate-pulse" />
                    </div>
                    <h2 className="text-2xl font-bold text-white mb-2 tracking-tight relative">
                        Waiting for webhook events...
                    </h2>
                    <p className="text-slate-400 mb-8 max-w-sm relative">
                        Your backend is running. Whenever a pipeline fails on GitHub, the webhook will instantly trigger the AI and appear here natively.
                    </p>
                </div>
            ) : (
                <>
                    {activeRuns.length > 0 && (
                        <div className="mb-2 z-0">
                            <div className="flex items-center gap-2.5 mb-4 px-1">
                                <span className="w-2 h-2 rounded-full animate-pulse bg-emerald-500 shadow-[0_0_6px_rgba(16,185,129,0.5)]" />
                                <span className="text-sm font-semibold text-white">Live</span>
                                <span className="text-xs text-label">{activeRuns.length} running</span>
                            </div>
                            <div className="grid grid-cols-1 gap-3">
                                <AnimatePresence mode="popLayout">
                                    {activeRuns.map((run, i) => (
                                        <RunCard key={run.id} run={run} index={i} />
                                    ))}
                                </AnimatePresence>
                            </div>
                        </div>
                    )}
                    <CompletedTable runs={completedRuns} />
                </>
            )}
        </div>
    );
}
