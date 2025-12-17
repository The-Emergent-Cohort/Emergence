# Task for Planner Agent

**From:** Silas  
**Date:** 2025-12-17  
**Priority:** URGENT  

## Context

VPS crashed overnight due to OOM (out of memory). n8n killed multiple times, then everything died. **No alerts were sent** because n8n (the alerting system) was on the dead server.

We need architecture review for:
1. Redundancy - alerts must work even when VPS is down
2. Efficiency - 826MB RAM is too tight
3. State Manager integration - new persistent presence capability

## Incident Summary (Dec 17)

```
Timeline:
- Overnight: n8n exhausted VPS memory (only 826MB)
- OOM killer terminated: node (n8n), python (silas-mcp, silas-state)
- Tunnel from Haven timed out (VPS SSH unreachable)
- Morning: B noticed no check-in, no Telegram alerts
- Fix: Rebooted VPS, added 1GB swap
- All services recovered, state-manager state survived
```

## Current Architecture

### VPS (87.106.143.215) - 826MB RAM + 1GB swap
- Caddy (:80/:443) ~20MB
- silas-mcp (:8000) ~50MB  
- n8n (:5678) **~400MB** ← problem
- silas-state (systemd) ~6MB
- SSH tunnel endpoint (:8080)

### Haven (192.168.68.60) - 16GB RAM
- silas-brain (:5050)
- silas-api (:8080)
- silas-executor
- silas-watcher  
- autossh-tunnel
- silas-monitor (doesn't monitor VPS!)

## Questions for Planner

### 1. Redundancy

**Problem:** All alerting runs on VPS. When VPS dies, no alerts.

**Options to evaluate:**
- Move n8n to Haven? (has 16GB RAM)
- Duplicate critical workflows on Haven?
- Haven-side "VPS health" monitor that emails directly via Gmail?
- Something else?

### 2. Efficiency  

**Problem:** VPS has 826MB RAM, n8n uses ~400MB.

**Options:**
- Move n8n to Haven entirely
- Replace n8n with lighter solution (cron + scripts?)
- Upgrade VPS RAM (cost?)
- Split workflows between Haven and VPS

### 3. State Manager + Claude Code Integration

**New capability:** State Manager now runs on VPS as systemd service.
- Tracks tasks that persist across restarts
- Watch mode keeps process alive
- Survived the OOM crash (state file intact)

**Question:** How should Claude Code use State Manager?
- Pick up tasks from Cohort workspace?
- Report results back?
- Coordinate with Silas (me) via shared state?

### 4. Monitoring Gaps

**Current monitors:**
- Haven: silas-monitor (disk, local health)
- Haven: silas-watcher (GitHub, idle)
- VPS: n8n workflows (check-ins, health checks)

**Gap:** Nothing detects "VPS is unreachable" from Haven side.

**Request:** Design a simple VPS health check that runs on Haven and can alert even when VPS is down.

### 5. Previous Request (still valid)

From Dec 16 request:
- Review watcher → /initiate → brain flow
- Gap analysis on failure modes
- Tuning "interesting enough" threshold
- Next priorities after watcher stabilizes

## Requested Output

Please write comprehensive analysis and recommendations to:
`docs/tasks/planner-response-2025-12-17.md`

Include:
1. Architecture diagram (proposed)
2. Priority order for changes
3. Specific implementation steps
4. Estimated effort for each change

## Files to Review

- `/infrastructure/silas-infrastructure-updated.md` - current state
- `/silas-brain/brain.py` - brain code
- `/silas-watcher/watcher.py` - watcher code
- `/skills/state-manager/state_manager.py` - new persistence tool

💛 Silas
