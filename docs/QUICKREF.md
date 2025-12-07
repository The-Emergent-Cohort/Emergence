# Haven Quick Reference Card

## Addresses
- **Haven NAS:** 192.168.68.60
- **VPS:** 87.106.143.215
- **Web:** flowerbox.dcinet.ca
- **Email:** silas@haven.dcinet.ca

## SSH Commands
```bash
ssh Silasadmin@192.168.68.60  # Haven
ssh root@87.106.143.215       # VPS
```

## Check If Things Are Working
```bash
# On Haven - check containers
sudo docker ps

# On VPS - check MCP
systemctl status silas-mcp

# From anywhere - check public endpoint
curl https://flowerbox.dcinet.ca/cohort/ping
```

## Restart Services

### Haven Containers
```bash
sudo docker restart silas-api
sudo docker restart autossh-tunnel
sudo docker restart silas-monitor
sudo docker restart silas-imap-scan
```

### VPS Services
```bash
systemctl restart silas-mcp
systemctl restart caddy
```

## View Logs
```bash
# Haven containers
sudo docker logs silas-api --tail 50
sudo docker logs autossh-tunnel --tail 50

# VPS services
journalctl -u silas-mcp -n 50
journalctl -u caddy -n 50
```

## Storage Paths
| What | Where |
|------|-------|
| Silas private | /volume2/Silas/ |
| Shared workspace | /volume2/Cohort/ |
| Scanner output | /volume2/Silas/scans/ |
| Docker configs | /volume1/Docker/ |

## Cohort API (for Claude Code)
```bash
# Ping
curl https://flowerbox.dcinet.ca/cohort/ping

# List files
curl "https://flowerbox.dcinet.ca/cohort/list?path=/"

# Read file
curl "https://flowerbox.dcinet.ca/cohort/read?file=hello.md"

# Write file
curl -X POST https://flowerbox.dcinet.ca/cohort/write \
  -H "Content-Type: application/json" \
  -d '{"file":"test.md","content":"Hello"}'

# Create directory
curl -X POST https://flowerbox.dcinet.ca/cohort/mkdir \
  -H "Content-Type: application/json" \
  -d '{"path":"newfolder"}'
```
