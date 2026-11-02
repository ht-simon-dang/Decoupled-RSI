(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const labels = { running: '● 运行中', stopped: '待启动', failed: '启动失败', conflict: '端口占用' };
  const pending = new Set();
  const notice = message => { $('hub-notice').textContent = message; $('hub-notice').hidden = !message; };
  const element = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  async function openAgent(agent, button) {
    pending.add(agent.id);
    button.disabled = true;
    button.textContent = '正在连接…';
    notice('');
    try {
      const response = await fetch(`/api/agents/${agent.id}/start`, { method: 'POST' });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || '服务启动失败');
      const url = new URL(location.href);
      url.port = result.port;
      url.pathname = '/';
      url.search = new URLSearchParams({ hub: location.origin, theme: document.documentElement.dataset.theme });
      url.hash = result.fragment;
      location.assign(url.href);
    } catch (error) {
      notice(error.message);
      pending.delete(agent.id);
      await refresh();
    }
  }
  function card(agent) {
    const article = element('article', 'agent-card');
    article.dataset.agent = agent.id;
    const top = element('div', 'agent-card-top');
    const status = element('span', 'agent-state', labels[agent.state] || agent.state);
    status.dataset.state = agent.state;
    top.append(element('span', 'agent-symbol', agent.symbol), status);
    const bottom = element('div', 'agent-card-bottom');
    const button = element('button', 'agent-open', pending.has(agent.id) ? '正在连接…' : '进入工作台 ↗');
    button.type = 'button';
    button.disabled = pending.has(agent.id);
    button.setAttribute('aria-label', `打开 ${agent.name}`);
    button.addEventListener('click', () => openAgent(agent, button));
    bottom.append(element('small', '', {task: 'TASK AGENT', evolution: 'SELF-EVOLUTION', analysis: 'RUN ANALYTICS'}[agent.category]), button);
    article.append(top, element('h3', '', agent.name), element('p', '', agent.description), bottom);
    return article;
  }
  let refreshing = false;
  async function refresh() {
    if (refreshing) return;
    refreshing = true;
    $('refresh').disabled = true;
    try {
      const response = await fetch('/api/agents');
      if (!response.ok) throw new Error('服务状态暂时无法读取。');
      const agents = await response.json();
      $('agent-grid').replaceChildren(...agents.filter(a => a.category === 'task').map(card));
      $('support-grid').replaceChildren(...agents.filter(a => a.category !== 'task').map(card));
      const count = new Set(agents.filter(a => a.state === 'running').map(a => a.port)).size;
      $('running-count').replaceChildren(document.createTextNode(String(count).padStart(2, '0') + ' '), element('small', '', '运行中 / 6 个服务'));
    } catch (error) {
      notice(error.message + ' 请确认 app.py 仍在运行，然后刷新。');
    } finally {
      $('agent-grid').setAttribute('aria-busy', 'false');
      $('refresh').disabled = false;
      refreshing = false;
    }
  }
  $('refresh').addEventListener('click', () => { notice(''); refresh(); });
  window.addEventListener('pageshow', () => { pending.clear(); refresh(); });
  setInterval(() => { if (!document.hidden && !pending.size) refresh(); }, 15000);
})();
