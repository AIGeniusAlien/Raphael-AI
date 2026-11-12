/* Real, opt-in browser speech services. No recorder, fixture or hidden upload. */
(() => {
  class RaphaelVoice {
    constructor({onText, onStatus, onCommand}) {
      this.onText = onText; this.onStatus = onStatus; this.onCommand = onCommand;
      this.recognition = null; this.active = false; this.timer = null; this.watchdog = null; this.serial = 0;
    }
    supported() { return !!(window.SpeechRecognition || window.webkitSpeechRecognition); }
    async start({consent, mode, wakePhrase, verifyConsent}) {
      if (!consent) throw new Error('Recording consent is required.');
      if (!this.supported()) throw new Error('Speech recognition is unavailable in this browser. Manual synthetic transcript entry remains available.');
      this.stop();
      const serial = this.serial;
      const media = await navigator.mediaDevices.getUserMedia({audio: true});
      media.getTracks().forEach(track => track.stop());
      if (serial !== this.serial) return;
      this.active = true;
      const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      this.recognition = new Recognition();
      const r = this.recognition;
      // Browsers may re-emit finalized result indices in subsequent events.
      // A repeated final must never create duplicate clinical transcripts or commands.
      const finalized = new Set();
      r.continuous = true; r.interimResults = true; r.lang = 'en-US';
      r.onspeechstart = () => { if (this.active && this.recognition === r) this.onStatus('Speech detected by the browser.'); };
      r.onspeechend = () => { if (this.active && this.recognition === r) this.onStatus('Waiting for speech.'); };
      r.onresult = event => {
        if (!this.active || this.recognition !== r) return;
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const result = event.results[i];
          const text = result[0].transcript.trim();
          if (finalized.has(i)) continue;
          if (!result.isFinal) { this.onStatus('Recognizing: ' + text); continue; }
          finalized.add(i);
          if (!text) continue;
          const phrase = wakePhrase.trim().toLowerCase();
          if (mode === 'handsfree' && phrase && text.toLowerCase().startsWith(phrase)
              && (text.length === phrase.length || /[,\s]/.test(text[phrase.length]))) {
            const command = text.slice(phrase.length).replace(/^[,\s]+/, '');
            // A wake command is an interaction request, never clinical dictation.
            // Suppress both empty wake-only results and automatic transcript writes.
            if (command) { window.speechSynthesis?.cancel(); this.onCommand(command); }
          } else {
            this.onText(text);
          }
        }
      };
      r.onerror = event => { if (this.recognition !== r) return; this.stop(); this.onStatus('Speech service error: ' + event.error + '. Microphone stopped.'); };
      r.onend = () => { if (this.recognition !== r) return; if (this.active && mode === 'handsfree') { try { finalized.clear(); r.start(); } catch (_) { this.stop(); } } else this.stop(); };
      try { r.start(); } catch (error) { this.stop(); throw error; }
      this.timer = setTimeout(() => { this.stop(); this.onStatus('Voice session timed out. Start again explicitly.'); }, 120000);
      if (typeof verifyConsent === 'function') {
        // Remote consent and encounter status can change while recognition
        // is running. Fail closed when this bounded periodic verification
        // cannot establish continuing authorization. This polling cannot
        // guarantee instantaneous remote revocation; the server separately
        // checks every transcript write.
        let verifying = false;
        this.watchdog = setInterval(async () => {
          if (verifying || !this.active || serial !== this.serial) return;
          verifying = true;
          let authorized = false;
          try { authorized = await verifyConsent() === true; } catch (_) {}
          verifying = false;
          if (!this.active || serial !== this.serial) return;
          if (!authorized) {
            this.stop();
            this.onStatus('Voice stopped: recording consent or encounter authorization could not be verified. Reauthorize before listening.');
          }
        }, 10000);
      }
      this.onStatus('Listening. Browser speech services may send audio to the browser vendor.');
    }
    stop() {
      this.serial += 1;
      this.active = false;
      clearTimeout(this.timer); this.timer = null;
      clearInterval(this.watchdog); this.watchdog = null;
      if (this.recognition) { this.recognition.onend = null; try { this.recognition.stop(); } catch (_) {} this.recognition = null; }
      window.speechSynthesis?.cancel();
    }
    speak(text) {
      if (!window.speechSynthesis || !window.SpeechSynthesisUtterance) return false;
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(new window.SpeechSynthesisUtterance(text));
      return true;
    }
  }
  window.RaphaelVoice = RaphaelVoice;
})();
