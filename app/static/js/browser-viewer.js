/**
 * BrowserViewer — renders Playwright screenshots on a canvas via WebSocket.
 *
 * Usage:
 *   const viewer = new BrowserViewer(canvasElement, wsUrl);
 *   viewer.on('cookies-extracted', ({ li_at, jsessionid, encrypted }) => { ... });
 *   viewer.on('profile-extracted', ({ first_name, last_name, public_id }) => { ... });
 *   viewer.on('error', ({ message }) => { ... });
 *   viewer.on('connected', () => { ... });
 *   viewer.on('disconnected', () => { ... });
 *   viewer.destroy();
 */
class BrowserViewer {
    constructor(canvas, wsUrl) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.ws = null;
        this.wsUrl = wsUrl;
        this._listeners = {};
        this._destroyed = false;

        // Bind event handlers
        this._onMouseDown = this._handleMouseDown.bind(this);
        this._onKeyDown = this._handleKeyDown.bind(this);
        this._onWheel = this._handleWheel.bind(this);

        canvas.addEventListener('mousedown', this._onMouseDown);
        canvas.addEventListener('wheel', this._onWheel, { passive: false });
        canvas.setAttribute('tabindex', '0');
        canvas.style.cursor = 'pointer';
        canvas.style.outline = 'none';
        canvas.addEventListener('keydown', this._onKeyDown);

        this._connect();
    }

    _connect() {
        if (this._destroyed) return;
        this.ws = new WebSocket(this.wsUrl);

        this.ws.onopen = () => {
            this._emit('connected');
        };

        this.ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                this._handleMessage(msg);
            } catch (e) {
                // binary or invalid
            }
        };

        this.ws.onclose = () => {
            this._emit('disconnected');
        };

        this.ws.onerror = () => {
            this._emit('error', { message: 'WebSocket connection error' });
        };
    }

    _handleMessage(msg) {
        switch (msg.type) {
            case 'screenshot':
                this._renderScreenshot(msg.data);
                break;
            case 'cookies':
                if (msg.status === 'ok') {
                    this._emit('cookies-extracted', {
                        li_at: msg.li_at,
                        jsessionid: msg.jsessionid,
                        encrypted: msg.encrypted
                    });
                } else {
                    this._emit('error', { message: msg.message || 'No LinkedIn cookies found. Please log in first.' });
                }
                break;
            case 'profile':
                if (msg.status === 'ok') {
                    this._emit('profile-extracted', {
                        first_name: msg.first_name,
                        last_name: msg.last_name,
                        public_id: msg.public_id,
                        profile_url: msg.profile_url
                    });
                } else {
                    this._emit('error', { message: msg.message || 'Could not extract profile.' });
                }
                break;
            case 'error':
                this._emit('error', { message: msg.message });
                break;
        }
    }

    _renderScreenshot(base64Data) {
        const img = new Image();
        img.onload = () => {
            this.canvas.width = img.width;
            this.canvas.height = img.height;
            this.ctx.drawImage(img, 0, 0);
        };
        img.src = 'data:image/jpeg;base64,' + base64Data;
    }

    _handleMouseDown(e) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;
        const x = Math.round((e.clientX - rect.left) * scaleX);
        const y = Math.round((e.clientY - rect.top) * scaleY);
        this.ws.send(JSON.stringify({ type: 'click', x, y }));
        this.canvas.focus();
    }

    _handleKeyDown(e) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
        e.preventDefault();

        // Special keys
        const specialKeys = {
            'Enter': 'Enter', 'Backspace': 'Backspace', 'Tab': 'Tab',
            'Escape': 'Escape', 'ArrowUp': 'ArrowUp', 'ArrowDown': 'ArrowDown',
            'ArrowLeft': 'ArrowLeft', 'ArrowRight': 'ArrowRight',
            'Delete': 'Delete', 'Home': 'Home', 'End': 'End',
        };

        if (specialKeys[e.key]) {
            this.ws.send(JSON.stringify({ type: 'key', key: specialKeys[e.key] }));
        } else if (e.key.length === 1) {
            this.ws.send(JSON.stringify({ type: 'type', text: e.key }));
        }
    }

    _handleWheel(e) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
        e.preventDefault();
        this.ws.send(JSON.stringify({ type: 'scroll', deltaY: Math.sign(e.deltaY) * 200 }));
    }

    extractCookies() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'extract_cookies' }));
        }
    }

    extractProfile() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'extract_profile' }));
        }
    }

    on(event, callback) {
        if (!this._listeners[event]) this._listeners[event] = [];
        this._listeners[event].push(callback);
    }

    _emit(event, data) {
        (this._listeners[event] || []).forEach(cb => cb(data));
    }

    destroy() {
        this._destroyed = true;
        this.canvas.removeEventListener('mousedown', this._onMouseDown);
        this.canvas.removeEventListener('wheel', this._onWheel);
        this.canvas.removeEventListener('keydown', this._onKeyDown);
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }
}
