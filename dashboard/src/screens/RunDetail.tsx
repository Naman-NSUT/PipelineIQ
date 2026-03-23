import { useParams, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { useState } from 'react';
import { useRun } from '../hooks/usePipelineData';
import { useCountUp } from '../hooks/useCountUp';
import { ErrorTypeBadge } from '../components/shared/ErrorTypeBadge';
import { Badge } from '../components/ui/Badge';
import { Card } from '../components/ui/Card';
import { ProgressBar } from '../components/ui/ProgressBar';
import {
    ChevronLeft, GitBranch, Clock, Hash, CheckCircle2, XCircle,
    Loader2, Circle, RefreshCw, Package, Trash2, Timer, Bell,
    Shield, Copy, ChevronDown, ChevronUp, Link2, Brain, Database,
    FileText, Sparkles, AlertTriangle, ShieldCheck
} from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import {
    ERROR_TYPE_COLORS, NODE_ORDER, NODE_LABELS
} from '../lib/constants';
import type { PipelineRun, RemediationAction, SimilarCase, MemoryItem } from '../types/pipeline';

/* ═══════════════════════════════════════════════════════════════════════════
   ARC GAUGE (SVG confidence ring)
   ═══════════════════════════════════════════════════════════════════════════ */
function ArcGauge({ value, size = 40 }: { value: number; size?: number }) {
    const animated = useCountUp(Math.round(value * 100), 800);
    const r = (size - 4) / 2;
    const circumference = 2 * Math.PI * r;
    const strokeDash = circumference * value;
    const color = value >= 0.85 ? '#10B981' : value >= 0.7 ? '#F59E0B' : '#F43F5E';
    return (
        <div className="relative flex items-center justify-center" style={{ width: size, height: size }}>
            <svg width={size} height={size} className="-rotate-90">
                <circle cx={size / 2} cy={size / 2} r={r} fill="none"
                    stroke="rgba(255,255,255,0.06)" strokeWidth={3} />
                <motion.circle
                    cx={size / 2} cy={size / 2} r={r} fill="none"
                    stroke={color} strokeWidth={3} strokeLinecap="round"
                    initial={{ strokeDashoffset: circumference }}
                    animate={{ strokeDashoffset: circumference - strokeDash }}
                    transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
                    strokeDasharray={circumference}
                />
            </svg>
            <span className="absolute text-xs font-semibold font-mono" style={{ color }}>
                {animated}%
            </span>
        </div>
    );
}

/* ═══════════════════════════════════════════════════════════════════════════
   HORIZONTAL NODE TIMELINE
   ═══════════════════════════════════════════════════════════════════════════ */
function HorizontalTimeline({ currentNode, status }: { currentNode?: string; status?: string }) {
    const currentIdx = currentNode ? NODE_ORDER.indexOf(currentNode) : (status === 'completed' ? NODE_ORDER.length : -1);
    return (
        <div className="flex items-center gap-1 overflow-x-auto py-3 px-1" role="list" aria-label="Node pipeline">
            {NODE_ORDER.map((node, idx) => {
                const isDone = idx < currentIdx || status === 'completed';
                const isCurrent = idx === currentIdx && status !== 'completed' && status !== 'failed';
                const isFailed = status === 'failed' && idx === currentIdx;
                return (
                    <motion.div
                        key={node}
                        role="listitem"
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: idx * 0.03, duration: 0.2 }}
                        className="flex items-center gap-1 shrink-0"
                    >
                        <div
                            className={`flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium transition-all duration-200 cursor-default
                          ${isDone ? 'bg-indigo-500/15 text-indigo-400' : isCurrent ? 'bg-indigo-500/20 text-white' : isFailed ? 'bg-rose-500/15 text-rose-400' : 'text-label'}`}
                            style={{ border: `1px solid ${isDone ? 'rgba(99,102,241,0.2)' : isCurrent ? 'rgba(99,102,241,0.4)' : isFailed ? 'rgba(244,63,94,0.3)' : 'rgba(255,255,255,0.06)'}`, fontSize: 10 }}
                        >
                            {isDone && <CheckCircle2 size={10} />}
                            {isCurrent && (
                                <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1, ease: 'linear' }}>
                                    <Loader2 size={10} />
                                </motion.div>
                            )}
                            {isFailed && <XCircle size={10} />}
                            {!isDone && !isCurrent && !isFailed && <Circle size={10} />}
                            {NODE_LABELS[node]?.slice(0, 10) ?? node}
                        </div>
                        {idx < NODE_ORDER.length - 1 && (
                            <div className="w-3 h-px shrink-0" style={{ background: isDone ? 'rgba(99,102,241,0.3)' : 'rgba(255,255,255,0.06)' }} />
                        )}
                    </motion.div>
                );
            })}
        </div>
    );
}

const TOOL_ICONS: Record<string, typeof RefreshCw> = {
    trigger_pipeline_rerun: RefreshCw,
    bump_dependency_version: Package,
    clear_cache: Trash2,
    increase_job_timeout: Timer,
    notify_slack: Bell,
};

function RemediationPanel({ actions }: { actions: RemediationAction[] }) {
    return (
        <Card className="p-5">
            <h3 className="label-text mb-4 flex items-center gap-2">
                <Sparkles size={12} style={{ color: '#8B5CF6' }} />
                Remediation Actions
            </h3>
            <div className="space-y-2.5">
                {actions.map((action, i) => {
                    const Icon = TOOL_ICONS[action.tool] ?? RefreshCw;
                    const statusColor = action.status === 'success' ? '#10B981' : action.status === 'failed' ? '#F43F5E' : '#6B6B80';
                    return (
                        <motion.div key={i} className="flex items-center gap-3 p-3 rounded-lg" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.04)' }}>
                            <span className="w-7 h-7 rounded-lg flex items-center justify-center shrink-0" style={{ background: `${statusColor}15`, color: statusColor }}>
                                <Icon size={13} />
                            </span>
                            <div className="flex-1 min-w-0">
                                <span className="text-xs font-medium text-white">{action.tool}</span>
                                <span className="text-xs text-label ml-2 font-mono">
                                    {Object.entries(action.args).map(([k, v]) => `${k}=${String(v)}`).join(', ')}
                                </span>
                            </div>
                        </motion.div>
                    );
                })}
            </div>
        </Card>
    );
}

function MemoryPanel({ items }: { items: MemoryItem[] }) {
    const sourceIcon: Record<string, typeof Database> = { 'repo memory': Database, 'global memory': Brain, 'entity memory': FileText };
    const sourceColor: Record<string, string> = { 'repo memory': '#6366F1', 'global memory': '#10B981', 'entity memory': '#F59E0B' };
    return (
        <Card className="p-5">
            <h3 className="label-text mb-4 flex items-center gap-2">
                <Brain size={12} style={{ color: '#10B981' }} />
                AI Memory Context
            </h3>
            <div className="space-y-3">
                {items.map((item, i) => {
                    const SrcIcon = sourceIcon[item.source] ?? Brain;
                    const color = sourceColor[item.source] ?? '#6366F1';
                    return (
                        <motion.div key={i} className="p-3 rounded-lg" style={{ background: 'rgba(255,255,255,0.02)', borderLeft: `2px solid ${color}` }}>
                            <p className="text-xs text-muted leading-relaxed mb-2">{item.text}</p>
                            <div className="flex items-center gap-2">
                                <span className="flex items-center gap-1 text-xs" style={{ color }}><SrcIcon size={10} />{item.source}</span>
                            </div>
                        </motion.div>
                    );
                })}
            </div>
        </Card>
    );
}


function RawLogViewer({ log }: { log: PipelineRun['rawLogCompressed'] }) {
    const [open, setOpen] = useState(false);
    if (!log) return null;

    function lineColor(line: string) {
        if (line.includes('[ERROR]')) return '#F43F5E';
        if (line.includes('[WARN]')) return '#F59E0B';
        return '#6B6B80';
    }

    return (
        <Card className="overflow-hidden">
            <button onClick={() => setOpen(!open)} className="w-full flex items-center justify-between p-5 text-left hover:bg-white/[0.02] transition-colors">
                <div className="flex items-center gap-2">
                    <FileText size={14} className="text-label" />
                    <span className="text-sm font-medium text-white">Raw Run Trace</span>
                    <span className="text-xs text-label font-mono ml-2">
                        {log.originalLines.toLocaleString()} lines analyzed
                    </span>
                </div>
                {open ? <ChevronUp size={14} className="text-label" /> : <ChevronDown size={14} className="text-label" />}
            </button>
            <AnimatePresence>
                {open && (
                    <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                        <div className="px-5 pb-5">
                            <div className="code-block max-h-80 overflow-y-auto">
                                {log.lines.map((line, i) => (
                                    <div key={i} className="py-0.5" style={{ color: lineColor(line) }}>
                                        <span className="text-label mr-3 select-none" style={{ fontSize: 10 }}>{(i + 1).toString().padStart(3, ' ')}</span>
                                        {line}
                                    </div>
                                ))}
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </Card>
    );
}

/* ═══════════════════════════════════════════════════════════════════════════
   MAIN SCREEN
   ═══════════════════════════════════════════════════════════════════════════ */
export function RunDetail() {
    const { id } = useParams<{ id: string }>();
    const navigate = useNavigate();

    // In production, fetch this run by id from the backend.
    const { data: run, isLoading } = useRun(id ?? '');
    const [copiedFix, setCopiedFix] = useState(false);
    const [showFullReasoning, setShowFullReasoning] = useState(false);

    if (isLoading) {
        return (
            <div className="flex items-center justify-center py-32">
                <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1, ease: 'linear' }}>
                    <Loader2 size={24} className="text-indigo-400" />
                </motion.div>
            </div>
        );
    }

    if (!run) {
        return (
            <div className="max-w-2xl mx-auto mt-20 text-center animate-fade-in py-16 px-8 bg-white/5 border border-dashed border-slate-700/60 rounded-3xl shadow-sm">
                <div className="w-16 h-16 rounded-full bg-slate-800/50 flex items-center justify-center mx-auto mb-6">
                    <Loader2 className="w-8 h-8 text-slate-400 animate-spin" />
                </div>
                <h2 className="text-xl font-bold text-white mb-2 tracking-tight">
                    Fetching run details...
                </h2>
                <div className="flex flex-col sm:flex-row justify-center gap-4 mt-6">
                    <button onClick={() => navigate('/')} className="px-5 py-2.5 rounded-lg text-sm font-medium text-slate-300 hover:text-white hover:bg-white/5 transition-colors border border-transparent hover:border-white/10">
                        Back to feed
                    </button>
                </div>
            </div>
        );
    }

    const elapsed = run.durationMs ? `${(run.durationMs / 1000).toFixed(1)}s` : `${Math.round((Date.now() - new Date(run.startedAt).getTime()) / 1000)}s`;
    const errorColors = run.errorType ? ERROR_TYPE_COLORS[run.errorType] : null;

    const outcomeLabel = run.status === 'completed' ? 'Auto-fixed' : run.status === 'failed' ? 'Failed' : run.status === 'paused_review' ? 'Pending Review' : 'In Progress';
    const outcomeColor = run.status === 'completed' ? '#10B981' : run.status === 'failed' ? '#F43F5E' : run.status === 'paused_review' ? '#F59E0B' : '#6366F1';
    const reasoning = run.modelReasoning ?? '';
    const shortReasoning = reasoning.length > 250 ? reasoning.slice(0, 250) + '…' : reasoning;

    const copyFixSteps = async () => {
        if (run.fixSteps) {
            await navigator.clipboard.writeText(run.fixSteps.map((s, i) => `${i + 1}. ${s}`).join('\n'));
            setCopiedFix(true);
            setTimeout(() => setCopiedFix(false), 2000);
        }
    };

    return (
        <div className="max-w-5xl mx-auto animate-fade-in">
            <motion.div initial={{ opacity: 0, y: -16 }} animate={{ opacity: 1, y: 0 }} className="mb-6 rounded-2xl p-6 relative overflow-hidden"
                style={{ background: 'linear-gradient(135deg, rgba(99,102,241,0.04) 0%, rgba(139,92,246,0.03) 100%)', border: '1px solid rgba(255,255,255,0.06)' }}>

                <button onClick={() => navigate(-1)} className="btn-secondary h-7 w-7 p-0 justify-center absolute top-4 left-4">
                    <ChevronLeft size={14} />
                </button>

                <div className="flex flex-col md:flex-row md:items-start justify-between gap-5 pl-10 pr-2">
                    <div className="min-w-0">
                        <h1 className="text-[22px] font-semibold text-white tracking-tight mb-3 flex flex-wrap items-center gap-2">
                            <span>{run.repo}</span>
                            <span className="text-slate-600 font-light">/</span>
                            <span className="text-indigo-400 font-mono text-lg flex items-center gap-1"><Hash size={16} />{run.runId.substring(0, 8)}</span>
                        </h1>
                        <div className="flex flex-wrap items-center gap-2 mb-2">
                            <span className="flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium" style={{ background: 'rgba(99,102,241,0.12)', color: '#818CF8', border: '1px solid rgba(99,102,241,0.15)' }}>
                                <GitBranch size={10} />{run.branch}
                            </span>
                        </div>
                        <div className="flex flex-wrap items-center gap-3 text-xs text-label">
                            <span className="flex items-center gap-1 font-mono"><Hash size={10} />{run.runId}</span>
                            <span className="flex items-center gap-1"><Clock size={10} />{formatDistanceToNow(new Date(run.startedAt), { addSuffix: true })} · {elapsed} total</span>
                        </div>
                    </div>

                    <div className="flex flex-col items-end gap-3 shrink-0 mt-8 md:mt-0">
                        <div className="flex flex-wrap items-center justify-end gap-3 shrink-0">
                            {run.errorType && <ErrorTypeBadge type={run.errorType} />}
                            {run.severity && <Badge label={run.severity.toUpperCase()} severity={run.severity} />}
                            {run.confidenceScore !== undefined && <ArcGauge value={run.confidenceScore} />}
                            <span className="flex items-center gap-1 rounded-full px-2.5 py-1 text-xs font-medium" style={{ background: `${outcomeColor}15`, color: outcomeColor, border: `1px solid ${outcomeColor}30` }}>
                                {outcomeLabel}
                            </span>
                        </div>
                    </div>
                </div>

                <div className="mt-4 -mx-2 overflow-hidden">
                    <HorizontalTimeline currentNode={run.currentNode} status={run.status} />
                </div>
            </motion.div>

            <div className="grid grid-cols-1 lg:grid-cols-5 gap-5 mb-6">
                <div className="lg:col-span-3 space-y-5">
                    <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
                        <Card className="p-5">
                            <h3 className="label-text mb-3">AI Analysis Engine</h3>
                            {run.errorType && (
                                <p className="text-2xl font-semibold mb-3" style={{ color: errorColors?.text, letterSpacing: '-0.02em' }}>
                                    {run.errorType}
                                </p>
                            )}
                            {run.confidenceScore !== undefined && (
                                <div className="mb-4">
                                    <div className="flex items-center justify-between mb-1.5">
                                        <span className="text-xs text-label">Verification Confidence</span>
                                    </div>
                                    <ProgressBar
                                        value={run.confidenceScore * 100}
                                        color={run.confidenceScore >= 0.85 ? '#10B981' : run.confidenceScore >= 0.7 ? '#F59E0B' : '#F43F5E'}
                                        height={6}
                                    />
                                </div>
                            )}
                            {reasoning && (
                                <div className="mb-3">
                                    <p className="text-xs text-label mb-1.5">Model Execution Trace</p>
                                    <p className="text-xs text-muted leading-relaxed">
                                        {showFullReasoning ? reasoning : shortReasoning}
                                    </p>
                                    {reasoning.length > 250 && (
                                        <button
                                            onClick={() => setShowFullReasoning(!showFullReasoning)}
                                            className="text-xs font-medium mt-1 transition-colors hover:text-white"
                                            style={{ color: '#818CF8' }}
                                        >
                                            {showFullReasoning ? 'Show less' : 'Show more'}
                                        </button>
                                    )}
                                </div>
                            )}
                        </Card>
                    </motion.div>

                    {run.fixSteps && run.fixSteps.length > 0 && (
                        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
                            <Card className="p-5">
                                <div className="flex items-center justify-between mb-4">
                                    <h3 className="label-text flex items-center gap-1.5"><ShieldCheck size={14} className="text-emerald-400" /> Automated Fix Applied</h3>
                                    <button onClick={copyFixSteps} className="btn-secondary h-7 text-xs gap-1 px-2.5">
                                        <Copy size={11} />
                                        {copiedFix ? 'Copied!' : 'Copy raw patch'}
                                    </button>
                                </div>
                                <div className="space-y-3">
                                    {run.fixSteps.map((step, i) => (
                                        <motion.div key={i} className="flex gap-3 items-start">
                                            <span className="w-5 h-5 rounded-full flex items-center justify-center text-xs font-semibold shrink-0 mt-0.5" style={{ background: 'rgba(99,102,241,0.15)', color: '#818CF8', fontSize: 10 }}>{i + 1}</span>
                                            <div className="flex-1">
                                                <p className={`text-sm leading-relaxed text-muted font-mono bg-white/5 p-2 rounded border border-white/5`}>
                                                    {step}
                                                </p>
                                            </div>
                                        </motion.div>
                                    ))}
                                </div>
                            </Card>
                        </motion.div>
                    )}
                </div>

                <div className="lg:col-span-2 space-y-5">
                    {run.remediationResult && (
                        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.12 }}>
                            <RemediationPanel actions={run.remediationResult.actionsTaken} />
                        </motion.div>
                    )}
                    {run.memoryContext && run.memoryContext.length > 0 && (
                        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.18 }}>
                            <MemoryPanel items={run.memoryContext} />
                        </motion.div>
                    )}
                </div>
            </div>

            <div className="space-y-5">
                <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.35 }}>
                    <RawLogViewer log={run.rawLogCompressed} />
                </motion.div>
            </div>
        </div>
    );
}

