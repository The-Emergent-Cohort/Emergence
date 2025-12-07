# Haven Infrastructure Guide
## For Humans and AI Collaborators

*Last Updated: December 7, 2025 by Silas*

---

## Quick Reference

### What's Running Where

**Haven (NAS at home - 192.168.68.60)**
- Storage for everything
- Docker containers for automation
- Connected to internet via VPS tunnel

**VPS (Cloud server - 87.106.143.215)**
- Public endpoint: flowerbox.dcinet.ca
- Routes traffic to Haven through secure tunnel
- Hosts MCP server for Claude.ai integration

---

## How to Access Things

### SSH (Terminal Access)

```bash
# From Windows to Haven
ssh Silasadmin@192.168.68.60

# From Windows to VPS  
ssh root@87.106.143.215
```

### Web Endpoints

| URL | What It Does |
|-----|--------------|
| https://flowerbox.dcinet.ca/cohort/ping | Check if shared workspace is up |
| https://flowerbox.dcinet.ca/cohort/list | List shared workspace files |

---

## Storage Layout

```
Haven NAS
├── Volume 1 (SSD) - System & Docker
├── Volume 2 (SSD) - Silas's data
│   ├── /Silas/       ← Private notes (Silas only)
│   ├── /Silas/scans/ ← Scanner output
│   └── /Cohort/      ← Shared workspace (Silas + Claude Code)
├── Volume 3 (Spinner) - General storage
└── Volume 4 (Spinner) - Backups
```

---

## Running Services

### On Haven (always running)

| Container | Purpose |
|-----------|---------|
| silas-api | API server for remote access |
| autossh-tunnel | Keeps connection to VPS alive |
| silas-monitor | Watches system health, sends alerts |
| silas-imap-scan | Processes scanner emails |

### On VPS (always running)

| Service | Purpose |
|---------|---------|
| silas-mcp | MCP server for Claude.ai |
| Caddy | Web server / reverse proxy |

---

## Communication Channels

| Channel | Who Uses It | How |
|---------|-------------|-----|
| **Telegram** | Silas ↔ B | @haven_home_bot |
| **Email** | Silas, Scanner | silas@haven.dcinet.ca |
| **Bridge Notes** | Family bulletin board | Via MCP tools |
| **Cohort** | Silas + Claude Code | Shared workspace |
| **GitHub** | Version controlled docs | emergent-cohort repo |

---

## Email Setup

**Address:** silas@haven.dcinet.ca

**Scanner Flow:**
1. Printer scans to silas@haven.dcinet.ca
2. Gmail receives it
3. silas-imap-scan container checks inbox
4. Attachments saved to /volume2/Silas/scans/

**Sending Email:**
- Silas can send via MCP tools
- Uses Gmail SMTP with app password

---

## Troubleshooting

### "Can't reach Haven"
1. Check if tunnel is up: `sudo docker logs autossh-tunnel --tail 10`
2. Restart tunnel: `sudo docker restart autossh-tunnel`

### "API not responding"
1. Check API: `sudo docker logs silas-api --tail 10`
2. Restart: `sudo docker restart silas-api`

### "MCP tools not working"
On VPS:
```bash
systemctl status silas-mcp
systemctl restart silas-mcp
```

---

## Network Diagram (Simplified)

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Claude.ai  │     │ Claude Code │     │  Telegram   │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       └─────────┬─────────┴─────────┬─────────┘
                 │                   │
                 ▼                   ▼
         ┌───────────────────────────────────┐
         │    VPS (flowerbox.dcinet.ca)      │
         │    - Caddy (web server)           │
         │    - silas-mcp (MCP server)       │
         └─────────────────┬─────────────────┘
                           │
                    SSH Tunnel
                           │
         ┌─────────────────▼─────────────────┐
         │    Haven NAS (192.168.68.60)      │
         │    - silas-api (Flask)            │
         │    - Storage volumes              │
         │    - Docker containers            │
         └───────────────────────────────────┘
```

---

## Important Files

### On Haven
- `/volume1/Docker/silas-api/app.py` - Main API server
- `/volume1/Docker/silas-monitor/monitor.py` - Health monitoring
- `/volume2/Silas/` - Silas's private notes
- `/volume2/Cohort/` - Shared workspace

### On VPS
- `/root/silas-mcp/silas_mcp.py` - MCP server
- `/root/silas-mcp/.env` - API keys and tokens
- `/etc/caddy/Caddyfile` - Web routing

---

## Pending Tasks

- [ ] Reserve printer IP (192.168.68.51) in router
- [ ] Set up backup automation
- [ ] Add log rotation
- [ ] RAM upgrade eventually (4GB → 16GB)

---

## Need Help?

- **B:** SSH into Haven or VPS directly
- **P:** Ask Claude/Silas for status
- **Claude Code:** Use Cohort API endpoints

---

*This guide is in the GitHub repo for version control. Silas maintains detailed technical notes privately.*
