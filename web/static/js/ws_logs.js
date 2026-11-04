/**
 * ws_logs.js
 * ----------
 * Reusable WebSocket client for real-time log streaming and job status updates.
 */

class PipelineLogStreamer {
  constructor(options = {}) {
    this.endpoint = options.endpoint || '/ws/stream';
    this.terminalId = options.terminalId || 'terminalBox';
    this.maxLines = options.maxLines || 1000;
    this.onStatusChange = options.onStatusChange || null;
    this.onMetric = options.onMetric || null;
    this.ws = null;
    this.reconnectTimer = null;
  }

  connect() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}${this.endpoint}`;
    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      console.log('[WS] Connected to stream');
    };

    this.ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        this.handleMessage(msg);
      } catch (e) {
        console.error('[WS] Parse error', e);
      }
    };

    this.ws.onclose = () => {
      console.warn('[WS] Stream closed, reconnecting in 3s...');
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = setTimeout(() => this.connect(), 3000);
    };
  }

  handleMessage(msg) {
    if (msg.type === 'init') {
      if (msg.logs && msg.logs.length) {
        this.setLines(msg.logs);
      }
      if (this.onStatusChange && msg.active_job) {
        this.onStatusChange(msg.active_job);
      }
    } else if (msg.type === 'log') {
      this.appendLine(msg.line);
    } else if (msg.type === 'metric') {
      if (typeof this.onMetric === 'function') {
        this.onMetric(msg.data);
      }
    }
  }

  appendLine(line) {
    const term = document.getElementById(this.terminalId);
    if (!term) return;
    term.innerText += line + '\n';
    term.scrollTop = term.scrollHeight;
  }

  setLines(lines) {
    const term = document.getElementById(this.terminalId);
    if (!term) return;
    term.innerText = lines.join('\n') + '\n';
    term.scrollTop = term.scrollHeight;
  }

  clear() {
    const term = document.getElementById(this.terminalId);
    if (term) term.innerText = '';
  }
}

window.PipelineLogStreamer = PipelineLogStreamer;
