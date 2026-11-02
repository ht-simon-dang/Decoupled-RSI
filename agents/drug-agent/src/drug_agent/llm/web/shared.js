/* Shared navigation and theme; no API interception or external dependencies. */
(() => {
  'use strict';
  const params = new URLSearchParams(location.search);
  const read = (key) => { try { return localStorage.getItem(key); } catch { return null; } };
  const save = (key, value) => { try { localStorage.setItem(key, value); } catch { /* private mode */ } };
  const validHub = (value) => {
    try {
      const url = new URL(value);
      return url.protocol === 'http:' && ['127.0.0.1', 'localhost'].includes(url.hostname) && !url.username && !url.password ? url.origin : null;
    } catch { return null; }
  };
  const hub = document.body.dataset.agentView === 'hub' ? location.origin
    : validHub(params.get('hub')) || validHub(read('rsi-hub')) || 'http://127.0.0.1:8080';
  save('rsi-hub', hub);
  let theme = params.get('theme') || read('rsi-theme') || 'light';
  if (!['light', 'dark'].includes(theme)) theme = 'light';
  const bar = document.createElement('div');
  bar.className = 'suite-bar';
  bar.setAttribute('role', 'navigation');
  bar.setAttribute('aria-label', 'Decoupled RSI');
  bar.innerHTML = '<a class="suite-brand">DECOUPLED <span>/ RSI</span></a><span class="suite-caption">AGENT WORKSPACE</span><div class="suite-actions"><a class="suite-home">← 所有 Agent</a><button type="button" class="suite-theme" aria-label="切换深浅主题"></button></div>';
  const themeButton = bar.querySelector('.suite-theme');
  const apply = (value) => {
    document.documentElement.dataset.theme = value;
    save('rsi-theme', value);
    const url = new URL(location.href);
    url.searchParams.set('theme', value);
    history.replaceState(history.state, '', url);
    themeButton.textContent = value === 'dark' ? '☀ 浅色' : '◐ 深色';
    themeButton.setAttribute('aria-pressed', String(value === 'dark'));
    bar.querySelectorAll('a').forEach(a => { a.href = hub + '/?theme=' + value; });
    window.dispatchEvent(new Event('rsi-theme-change'));
  };
  apply(theme);
  themeButton.addEventListener('click', () => apply(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'));
  document.body.prepend(bar);
  window.addEventListener('storage', event => {
    if (event.key === 'rsi-theme' && ['light', 'dark'].includes(event.newValue)) apply(event.newValue);
  });
  // The Self-Evolve entry shares the existing console service and its tab.
  if (document.body.dataset.agentView === 'drug' && location.hash === '#evolve') {
    document.querySelector('#viewtabs button[data-val="evolve"]')?.click();
  }
})();
