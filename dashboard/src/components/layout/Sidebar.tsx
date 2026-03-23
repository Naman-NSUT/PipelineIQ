import { useState, useEffect } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Activity, FileSearch, Cpu } from 'lucide-react';

const MAIN_NAV = [
    { to: '/', icon: Activity, label: 'Live feed' },
    { to: '/runs', icon: FileSearch, label: 'Runs' },
];

export function Sidebar() {
    // Active link styles
    const getLinkClass = (isActive: boolean) => {
        return `group flex items-center gap-3 px-3 py-3 rounded-lg text-sm font-medium transition-all duration-150 relative overflow-hidden ${isActive
            ? 'bg-indigo-500/10 text-white'
            : 'text-slate-400 hover:bg-white/[0.04] hover:text-white'
            }`;
    };

    const getIconClass = (isActive: boolean) => {
        return `shrink-0 transition-colors ${isActive ? 'text-indigo-400' : 'text-slate-500 group-hover:text-slate-400'}`;
    };

    return (
        <>
            {/* ── DESKTOP SIDEBAR ─────────────────────────────────────────────────── */}
            <aside className="hidden md:flex flex-col h-screen w-[220px] fixed left-0 top-0 z-40 bg-[#111118] border-r border-white/5 shadow-[4px_0_24px_rgba(0,0,0,0.2)]">

                {/* 1. Header (Logo) */}
                <div className="pt-6 px-4 pb-4 border-b border-white/5 mx-2">
                    <div className="flex items-center gap-2.5 h-10 mb-2">
                        <div className="w-7 h-7 rounded bg-indigo-500/20 flex items-center justify-center border border-indigo-500/30">
                            <Cpu size={14} className="text-indigo-400" />
                        </div>
                        <span className="text-base font-bold tracking-tight text-white drop-shadow-sm">PipelineIQ</span>
                    </div>
                </div>

                {/* 2. Main Navigation */}
                <div className="flex-1 overflow-y-auto overflow-x-hidden scrollbar-hidden px-3 py-6 space-y-6">

                    {/* Core Routes */}
                    <nav className="space-y-1" aria-label="Main Navigation">
                        {MAIN_NAV.map(({ to, icon: Icon, label }) => (
                            <NavLink key={to} to={to} end={to === '/'} className={({ isActive }) => getLinkClass(isActive)}>
                                {({ isActive }) => (
                                    <>
                                        {isActive && <div className="absolute left-0 top-0 bottom-0 w-[3px] bg-indigo-500 rounded-r-full shadow-[0_0_8px_rgba(99,102,241,0.8)]" />}
                                        <Icon size={18} className={getIconClass(isActive)} />
                                        <span className="flex-1 truncate leading-none pt-px">{label}</span>
                                    </>
                                )}
                            </NavLink>
                        ))}
                    </nav>

                </div>
            </aside>

            {/* ── MOBILE BOTTOM TAB BAR ───────────────────────────────────────────── */}
            <div className="md:hidden fixed bottom-0 left-0 right-0 z-40 bg-[#111118]/90 backdrop-blur-lg border-t border-white/10 pb-safe">
                <div className="flex items-center justify-around h-16 px-2">
                    {MAIN_NAV.map(({ to, icon: Icon, label }) => (
                        <NavLink key={to} to={to} end={to === '/'} className={({ isActive }) => `flex flex-col items-center justify-center w-full h-full gap-1 relative transition-colors ${isActive ? 'text-indigo-400' : 'text-slate-400 hover:text-white'}`}>
                            {({ isActive }) => (
                                <>
                                    {isActive && <motion.div layoutId="mobile-indicator" className="absolute top-0 w-8 h-0.5 bg-indigo-500 rounded-b-full shadow-[0_0_8px_rgba(99,102,241,0.8)]" />}
                                    <Icon size={20} />
                                    <span className="text-[10px] font-medium leading-none">{label}</span>
                                </>
                            )}
                        </NavLink>
                    ))}
                </div>
            </div>
        </>
    );
}
