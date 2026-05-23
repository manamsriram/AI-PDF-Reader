document.addEventListener('alpine:init', () => {
  Alpine.data('docsenseApp', () => ({
    messages: [],
    sources: [],
    documents: [],
    question: '',
    loading: false,
    docsExpanded: false,
    uploadStatus: '',
    uploadStatusType: '',

    async init() {
      await this.loadDocuments();
    },

    async loadDocuments() {
      try {
        const res = await fetch('/documents');
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
        const res = await fetch('/upload', { method: 'POST', body: formData });
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

      // Reset textarea height
      const ta = this.$refs.textarea;
      if (ta) { ta.style.height = 'auto'; }

      await this.$nextTick();
      this._scrollToBottom();

      try {
        const res = await fetch('/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
          body: 'question=' + encodeURIComponent(q),
        });

        if (!res.ok) throw new Error('Network error');

        const data = await res.json();
        const text = data.response || "I couldn't find an answer.";

        this.messages.push({
          role: 'bot',
          text,
          html: marked.parse(text),
        });

        this.sources = data.sources || [];
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
