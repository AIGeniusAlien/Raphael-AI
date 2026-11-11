/* Keyboard-accessible conversation composer; no clinical data leaves this module. */
(() => {
  'use strict';
  function install(form, textarea) {
    if (!form || !textarea || typeof form.requestSubmit !== 'function') return false;
    function resize() {
      // Keep the input comfortable without displacing the conversation.
      textarea.style.height = 'auto';
      const height = Math.max(80, Math.min(textarea.scrollHeight || 80, 240));
      textarea.style.height = height + 'px';
    }
    textarea.addEventListener('input', resize);
    textarea.addEventListener('keydown', event => {
      if (event.key !== 'Enter' || event.shiftKey || event.isComposing ||
          event.ctrlKey || event.metaKey || event.altKey || event.keyCode === 229) return;
      if (!textarea.value.trim() || form.querySelector('#send')?.disabled) return;
      event.preventDefault();
      form.requestSubmit();
    });
    resize();
    return true;
  }
  window.RaphaelComposer = {install};
})();
