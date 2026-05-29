let _supabase = null;
try {
  const _cfg = document.getElementById('supabase-config');
  if (_cfg && _cfg.dataset.url) {
    _supabase = supabase.createClient(_cfg.dataset.url, _cfg.dataset.key);
  }
} catch (e) {
  console.error('Supabase client init failed:', e);
}

document.addEventListener('alpine:init', () => {
  Alpine.data('docsenseApp', () => ({
    // Auth
    authToken: localStorage.getItem('sb_token') || null,
    authMode: 'login',
    authEmail: '',
    authPassword: '',
    authError: '',
    authLoading: false,

    // App
    messages: [],
    sources: [],
    documents: [],
    history: [],
    question: '',
    loading: false,
    docsExpanded: false,
    uploadStatus: '',
    uploadStatusType: '',
    sidebarOpen: true,

    async init() {
      if (!this.authToken) return;
      // Validate token by calling an auth-protected endpoint
      try {
        const res = await fetch('/history', { headers: this.authHeaders() });
        if (res.status === 401) {
          this.authToken = null;
          localStorage.removeItem('sb_token');
          return;
        }
        const data = await res.json();
        this.history = data.history || [];
      } catch (e) {
        console.error('Init check failed:', e);
      }
      await this.loadDocuments();
    },

    authHeaders() {
      return this.authToken ? { 'Authorization': `Bearer ${this.authToken}` } : {};
    },

    async login() {
      this.authError = '';
      if (!_supabase) { this.authError = 'Auth not configured. Check server env vars.'; return; }
      this.authLoading = true;
      try {
        const { data, error } = await _supabase.auth.signInWithPassword({
          email: this.authEmail,
          password: this.authPassword,
        });
        if (error) { this.authError = error.message; return; }
        this.authToken = data.session.access_token;
        localStorage.setItem('sb_token', this.authToken);
        await this.loadDocuments();
        await this.loadHistory();
      } catch (e) {
        this.authError = 'Sign in failed. Please try again.';
      } finally {
        this.authLoading = false;
      }
    },

    async signup() {
      this.authError = '';
      if (!_supabase) { this.authError = 'Auth not configured. Check server env vars.'; return; }
      this.authLoading = true;
      try {
        const { data, error } = await _supabase.auth.signUp({
          email: this.authEmail,
          password: this.authPassword,
        });
        if (error) { this.authError = error.message; return; }
        if (data.session) {
          this.authToken = data.session.access_token;
          localStorage.setItem('sb_token', this.authToken);
          await this.loadDocuments();
          await this.loadHistory();
        } else {
          this.authError = 'Check your email to confirm your account, then sign in.';
        }
      } catch (e) {
        this.authError = 'Sign up failed. Please try again.';
      } finally {
        this.authLoading = false;
      }
    },

    async logout() {
      if (_supabase) await _supabase.auth.signOut();
      this.authToken = null;
      localStorage.removeItem('sb_token');
      this.messages = [];
      this.sources = [];
      this.documents = [];
      this.history = [];
      this.question = '';
    },

    async loadHistory() {
      try {
        const res = await fetch('/history', { headers: this.authHeaders() });
        if (!res.ok) return;
        const data = await res.json();
        this.history = data.history || [];
      } catch (e) {
        console.error('Failed to load history:', e);
      }
    },

    loadHistoryItem(item) {
      this.messages = [
        { role: 'user', text: item.question },
        { role: 'bot', text: item.answer, html: marked.parse(item.answer) },
      ];
      this.sources = item.sources || [];
      this.$nextTick(() => this._scrollToBottom());
    },

    formatDate(iso) {
      return new Date(iso).toLocaleDateString(undefined, {
        month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
      });
    },

    async loadDocuments() {
      try {
        const res = await fetch('/documents', { headers: this.authHeaders() });
        const data = await res.json();
        this.documents = data.documents || [];
      } catch (e) {
        console.error('Failed to load documents:', e);
      }
    },

    toggleDocs() {
      this.docsExpanded = !this.docsExpanded;
    },

    get docCount() {
      return this.documents.length;
    },

    triggerUpload() {
      this.$refs.fileInput.click();
    },

    async handleUpload(event) {
      const file = event.target.files[0];
      if (!file) return;

      this.docsExpanded = true;
      this.uploadStatus = `Indexing ${file.name}…`;
      this.uploadStatusType = '';

      const formData = new FormData();
      formData.append('pdf', file);

      try {
        const res = await fetch('/upload', {
          method: 'POST',
          headers: this.authHeaders(),
          body: formData,
        });
        const data = await res.json();
        if (data.error) {
          this.uploadStatus = `Error: ${data.error}`;
          this.uploadStatusType = 'error';
        } else {
          this.uploadStatus = `✓ ${data.message}`;
          this.uploadStatusType = 'success';
          await this.loadDocuments();
        }
      } catch (e) {
        this.uploadStatus = 'Upload failed. Please try again.';
        this.uploadStatusType = 'error';
      }

      event.target.value = '';
      setTimeout(() => { this.uploadStatus = ''; this.uploadStatusType = ''; }, 4000);
    },

    async ask() {
      const q = this.question.trim();
      if (!q || this.loading) return;

      this.messages.push({ role: 'user', text: q });
      this.question = '';
      this.loading = true;
      this.sources = [];

      const ta = this.$refs.textarea;
      if (ta) { ta.style.height = 'auto'; }

      await this.$nextTick();
      this._scrollToBottom();

      try {
        const res = await fetch('/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded', ...this.authHeaders() },
          body: 'question=' + encodeURIComponent(q),
        });

        if (!res.ok) throw new Error('Network error');

        const data = await res.json();
        const text = data.response || "I couldn't find an answer.";

        this.messages.push({ role: 'bot', text, html: marked.parse(text) });
        this.sources = data.sources || [];
        this.loadHistory();
      } catch (e) {
        const fallback = 'Something went wrong, please try again.';
        this.messages.push({ role: 'bot', text: fallback, html: fallback });
      } finally {
        this.loading = false;
        await this.$nextTick();
        this._scrollToBottom();
      }
    },

    handleKeydown(event) {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        this.ask();
      }
    },

    autoResize(event) {
      const el = event.target;
      el.style.height = 'auto';
      el.style.height = Math.min(el.scrollHeight, 120) + 'px';
    },

    scorePercent(score) {
      return Math.round(score * 100) + '%';
    },

    _scrollToBottom() {
      const el = this.$refs.messagesList;
      if (el) el.scrollTop = el.scrollHeight;
    },
  }));
});
