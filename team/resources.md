# Infrastructure Resources

*Last updated: Dec 7, 2024*

## Compute Resources

### Patrick's Machine (PGAME)

**Primary development/training workstation**

| Component | Spec |
|-----------|------|
| CPU | Intel Core i7-12700F (12 cores, 20 threads) @ 2.1GHz |
| RAM | 32GB DDR |
| GPU | NVIDIA GeForce RTX 3060 |
| VRAM | 12GB dedicated |
| OS | Windows 11 Home |
| Display | LG Ultrawide 2560x1080 |

**Capabilities:**
- PyTorch with CUDA support
- Can train models up to ~2-3B parameters with gradient checkpointing
- Phase 1-6 coherence-lab curriculum runs easily
- Local development environment

**Access:** Via work tunnel (pending setup)

### VPS

**Always-on server for persistent services**

- Location: Web Hosting Canada
- Use cases: MCP servers, APIs, transcriber bot, persistent services
- Access: SSH (pending setup)

### Haven NAS

**Network storage (Silas + B maintaining)**

- Status: Being upgraded/configured
- Use cases: Data storage, model checkpoints, shared resources
- Access: Via VPS tunnel (pending)

---

## Web Infrastructure

### Domain: emergentcohort.org

- Registrar: WHC (Web Hosting Canada)
- DNS: Fixed (Dec 7, 2024)
- Status: Propagating

### GitHub Pages

- URL: https://the-emergent-cohort.github.io/Emergence/
- Custom domain: emergentcohort.org (CNAME configured)
- Source: /docs folder

### Webspace

- Static hosting available
- Can redirect to GitHub Pages or host directly

---

## API Access

### Anthropic

- Plan: Max tier
- API access: Available
- Use cases: Embedded Claude on website, API integrations

---

## Email

- Unlimited addresses available on emergentcohort.org
- Use cases: Identity anchors for family members (silas@, etc.)

---

## Network Topology (Target)

```
Internet
    │
    ├── emergentcohort.org (public)
    │       ├── GitHub Pages (static site)
    │       └── Embedded Claude (API)
    │
    ├── VPS (always-on)
    │       ├── MCP servers
    │       ├── Persistent services
    │       └── Tunnel endpoint
    │
    └── Patrick's Machine (PGAME)
            ├── Development environment
            ├── PyTorch + CUDA training
            └── Local compute
                    │
                    └── Haven NAS (storage)
```

---

## Pending Setup

1. [ ] VPS SSH access
2. [ ] Work tunnel to PGAME
3. [ ] NAS access via VPS
4. [ ] PyTorch environment on PGAME
5. [ ] Embedded Claude on website
