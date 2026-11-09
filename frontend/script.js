// SmartBank frontend: cookie-authenticated calls shared by all pages.
const isLiveServer = ['5500', '5501', '5502'].includes(window.location.port);
const API_BASE_URL = isLiveServer
  ? `http://${window.location.hostname === 'localhost' ? 'localhost' : '127.0.0.1'}:8000`
  : (window.location.origin && window.location.origin.startsWith('http') ? window.location.origin : 'http://127.0.0.1:8000');
const money = (value) => new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(Number(value || 0));
const shortDate = (value) => value ? new Intl.DateTimeFormat([], { dateStyle: 'medium' }).format(new Date(value)) : '—';

async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      credentials: 'include',
      headers: { ...(options.body && !(options.body instanceof FormData) ? { 'Content-Type': 'application/json' } : {}), ...options.headers },
    });
  } catch (_) {
    throw new Error(`Can't reach SmartBank's API at ${API_BASE_URL}. Start it from the project folder with: uvicorn backend.main:app --reload. Open this page through Live Server, not as a file.`);
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map((item) => item.msg).join(', ') : data.detail;
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return data;
}

function messageFor(element, message, success = false) {
  if (!element) return;
  element.textContent = message;
  element.classList.toggle('success', success);
}

function formPayload(form) {
  const payload = Object.fromEntries(new FormData(form).entries());
  for (const [key, value] of Object.entries(payload)) {
    if (form.elements[key]?.type === 'number') payload[key] = Number(value);
    if (key === 'international_transaction') payload[key] = value === 'true';
  }
  return payload;
}

async function setupAuthForms() {
  const login = document.getElementById('login-form');
  if (login) login.addEventListener('submit', async (event) => {
    event.preventDefault();
    const button = login.querySelector('button[type="submit"]');
    button.disabled = true;
    const target = document.getElementById('login-message');
    messageFor(target, 'Signing you in…');
    try {
      await api('/api/auth/login', { method: 'POST', body: JSON.stringify(formPayload(login)) });
      window.location.href = 'dashboard.html';
    } catch (error) { messageFor(target, error.message); }
    finally { button.disabled = false; }
  });

  const register = document.getElementById('register-form');
  if (register) register.addEventListener('submit', async (event) => {
    event.preventDefault();
    const button = register.querySelector('button[type="submit"]');
    button.disabled = true;
    const target = document.getElementById('register-message');
    messageFor(target, 'Creating your account…');
    try {
      await api('/api/auth/register', { method: 'POST', body: JSON.stringify(formPayload(register)) });
      messageFor(target, 'Account created. You can now sign in.', true);
      window.setTimeout(() => { window.location.href = 'login.html'; }, 900);
    } catch (error) { messageFor(target, error.message); }
    finally { button.disabled = false; }
  });
}

async function loadApplications() {
  const data = await api('/api/loan/my-applications');
  return data.applications;
}

function applicationTable(applications, compact = false) {
  if (!applications.length) return '<p class="muted">No loan applications yet. Start an application to see it here.</p>';
  const rows = applications.slice(0, compact ? 5 : applications.length).map((item) => `
    <tr><td><a class="table-link" href="application-details.html?id=${encodeURIComponent(item.application_id)}">${item.application_id}</a></td>
    <td>${shortDate(item.application_date)}</td><td>${money(item.requested_loan_amount)}</td>
    <td><span class="risk-badge ${String(item.risk_category).toLowerCase()}">${item.risk_category} risk</span></td>
    <td>${item.application_status}</td></tr>`).join('');
  return `<table class="data-table"><thead><tr><th>Reference</th><th>Date</th><th>Requested</th><th>Risk estimate</th><th>Status</th></tr></thead><tbody>${rows}</tbody></table>`;
}

async function setupCustomerPages(user) {
  const welcome = document.getElementById('welcome-title');
  if (welcome) welcome.textContent = `Welcome back, ${user.name.split(' ')[0]}`;
  const profileName = document.getElementById('profile-name');
  if (profileName) {
    profileName.textContent = user.name;
    document.getElementById('profile-email').textContent = user.email;
    document.getElementById('profile-role').textContent = user.role;
    document.getElementById('profile-initial').textContent = user.name.trim().charAt(0).toUpperCase();
  }
  const applicantName = document.getElementById('applicant-name');
  if (applicantName) applicantName.value = user.name;

  const dashboard = document.getElementById('dashboard-applications');
  if (document.getElementById('account-balance')) {
    try {
      const { account, credit_score, recent_transactions } = await api('/api/account/summary');
      document.getElementById('account-balance').textContent = money(account.available_balance);
      document.getElementById('account-number').textContent = account.masked_account_number;
      document.getElementById('account-income').textContent = money(account.monthly_income);
      document.getElementById('account-expenses').textContent = money(account.monthly_expenses);
      const scoreValue = typeof credit_score === 'object' && credit_score !== null ? credit_score.score : credit_score;
      const categoryValue = (typeof credit_score === 'object' && credit_score !== null ? credit_score.category : null) || 'Illustrative';
      document.getElementById('account-credit').textContent = scoreValue ?? '—';
      document.getElementById('account-credit-category').textContent = `${categoryValue} · illustrative score`;
      const recent = document.getElementById('dashboard-transactions');
      if (recent) recent.innerHTML = `<table class="data-table"><thead><tr><th>Date</th><th>Description</th><th>Category</th><th>Amount</th></tr></thead><tbody>${recent_transactions.map(t=>`<tr><td>${escapeHTML(shortDate(t.date))}</td><td>${escapeHTML(t.description)}</td><td>${escapeHTML(t.merchant_category)}</td><td>${t.transaction_type === 'Credit' ? '+' : '−'}${money(t.amount)}</td></tr>`).join('')}</tbody></table>`;
    } catch (error) { document.getElementById('account-balance').textContent = error.message; }
  }
  const list = document.getElementById('applications-list');
  if (dashboard || list) {
    try {
      const applications = await loadApplications();
      const total = applications.reduce((sum, item) => sum + Number(item.requested_loan_amount), 0);
      if (dashboard) {
        document.getElementById('dashboard-count').textContent = String(applications.length);
        document.getElementById('dashboard-total').textContent = money(total);
        document.getElementById('dashboard-review').textContent = String(applications.filter((item) => item.application_status === 'Under Review').length);
        dashboard.innerHTML = applicationTable(applications, true);
      }
      if (list) list.innerHTML = applicationTable(applications);
    } catch (error) {
      if (dashboard) dashboard.textContent = error.message;
      if (list) list.textContent = error.message;
    }
  }
}

async function setupBankingPages() {
  const table = document.getElementById('transactions-table');
  const search = document.getElementById('transaction-search');
  const category = document.getElementById('transaction-category');
  const kind = document.getElementById('transaction-type');
  const sort = document.getElementById('transaction-sort');
  async function loadTransactions() {
    const params = new URLSearchParams();
    if (search?.value) params.set('search', search.value);
    if (category?.value) params.set('category', category.value);
    if (kind?.value) params.set('type', kind.value);
    if (sort?.value) params.set('sort', sort.value);
    const { transactions } = await api(`/api/transactions?${params}`);
    table.innerHTML = transactions.length ? `<table class="data-table"><thead><tr><th>Date</th><th>Description</th><th>Category</th><th>Type</th><th>Amount</th><th>Status</th></tr></thead><tbody>${transactions.map(t=>`<tr><td>${escapeHTML(shortDate(t.date))}</td><td>${escapeHTML(t.description)}</td><td>${escapeHTML(t.merchant_category)}</td><td>${escapeHTML(t.transaction_type)}</td><td>${money(t.amount)}</td><td>${escapeHTML(t.status)}</td></tr>`).join('')}</tbody></table>` : '<p class="muted">No transactions match these filters.</p>';
  }
  if (table) {
    try { await loadTransactions(); } catch (error) { table.textContent = error.message; }
    [search, category, kind, sort].filter(Boolean).forEach(el => el.addEventListener(el === search ? 'input' : 'change', () => loadTransactions().catch(e => { table.textContent = e.message; })));
  }
  const credit = document.getElementById('credit-profile');
  if (credit) {
    try { const { profile } = await api('/api/credit/profile'); credit.innerHTML = `<div class="metric-grid"><article class="metric-card"><span>Demo score</span><strong>${profile.score}</strong><small>${escapeHTML(profile.category)}</small></article><article class="metric-card"><span>Credit utilization</span><strong>${profile.utilization_percent}%</strong><small>Illustrative profile</small></article><article class="metric-card"><span>Monthly EMI</span><strong>${money(profile.monthly_emi)}</strong><small>Profile estimate</small></article><article class="metric-card"><span>Debt to income</span><strong>${(profile.debt_to_income_ratio*100).toFixed(1)}%</strong><small>Monthly EMI / income</small></article></div><p class="muted">${escapeHTML(profile.insights.join(' '))} This is synthetic demonstration data, not a bureau record.</p>`; }
    catch (error) { credit.textContent = error.message; }
  }
  const analytics = document.getElementById('spending-analytics');
  if (analytics) {
    try { const d = await api('/api/transactions/analytics'); analytics.innerHTML = `<div class="metric-grid"><article class="metric-card"><span>Total income</span><strong>${money(d.total_income)}</strong></article><article class="metric-card"><span>Expenditure</span><strong>${money(d.total_expenditure)}</strong></article><article class="metric-card"><span>Net savings</span><strong>${money(d.net_savings)}</strong></article><article class="metric-card"><span>Average transaction</span><strong>${money(d.average_transaction_amount)}</strong></article></div><h2>Expenses by category</h2><div class="table-wrap"><table class="data-table"><thead><tr><th>Category</th><th>Amount</th></tr></thead><tbody>${d.category_expenses.map(x=>`<tr><td>${escapeHTML(x.category)}</td><td>${money(x.amount)}</td></tr>`).join('')}</tbody></table></div>`; }
    catch (error) { analytics.textContent = error.message; }
  }
  const alerts = document.getElementById('fraud-alerts');
  if (alerts) { try { const { alerts: items } = await api('/api/fraud/alerts'); alerts.innerHTML = items.length ? items.map(a=>`<article class="content-card"><strong>${escapeHTML(a.risk_level)} risk alert</strong><p>${escapeHTML(a.explanation)}</p><small>${escapeHTML(a.date)}</small></article>`).join('') : '<p class="muted">No demo alerts for this account.</p>'; } catch(e) { alerts.textContent = e.message; } }
}

function setupLoanApplication() {
  const form = document.getElementById('application-form');
  if (!form) return;
  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const button = form.querySelector('button[type="submit"]');
    const target = document.getElementById('application-message');
    button.disabled = true;
    messageFor(target, 'Submitting your application…');
    try {
      const data = await api('/api/loan/apply', { method: 'POST', body: JSON.stringify(formPayload(form)) });
      messageFor(target, `Application ${data.application_id} saved. ${data.risk_category} risk estimate · ${data.application_status}.`, true);
      form.reset();
      window.setTimeout(() => { window.location.href = 'applications.html'; }, 1800);
    } catch (error) { messageFor(target, error.message); }
    finally { button.disabled = false; }
  });
}

async function setupApplicationDetails() {
  const root = document.getElementById('application-detail');
  if (!root) return;
  const id = new URLSearchParams(window.location.search).get('id');
  if (!id) { root.textContent = 'Application not found.'; return; }
  try {
    const { application: item } = await api(`/api/loan/applications/${encodeURIComponent(id)}`);
    root.innerHTML = `<p class="eyebrow">APPLICATION ${item.application_id}</p><h2>${item.application_status}</h2><div class="detail-grid">
      <span>Applicant</span><strong>${escapeHTML(item.applicant_name)}</strong><span>Submitted</span><strong>${shortDate(item.application_date)}</strong>
      <span>Requested amount</span><strong>${money(item.requested_loan_amount)}</strong><span>Risk estimate</span><strong>${item.risk_category} · ${Number(item.risk_score).toFixed(1)}/100</strong>
      <span>Model probability</span><strong>${(Number(item.loan_probability) * 100).toFixed(1)}%</strong><span>Credit score</span><strong>${item.credit_score}</strong>
      <span>Annual income</span><strong>${money(item.annual_income)}</strong><span>Employment</span><strong>${escapeHTML(item.employment_type)}</strong></div>
      <p class="muted">This experimental estimate does not represent an approval or lending decision.</p>`;
  } catch (error) { root.textContent = error.message; }
}

function escapeHTML(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);
}

function setupLogout() {
  document.querySelectorAll('[data-logout]').forEach((button) => button.addEventListener('click', async () => {
    button.disabled = true;
    try { await api('/api/auth/logout', { method: 'POST' }); } catch (_) { /* The session may already be expired. */ }
    window.location.href = 'login.html';
  }));
}

async function setupProtectedPage() {
  if (!document.body.dataset.protected) return null;
  let data;
  try { data = await api('/api/auth/me'); }
  catch (_) {
    const next = encodeURIComponent(`${window.location.pathname.split('/').pop()}${window.location.search}`);
    window.location.replace(`login.html?next=${next}`);
    return null;
  }
  const user = data.user;
  document.querySelectorAll('[data-admin-only]').forEach((item) => { item.hidden = user.role !== 'admin'; });
  if (document.body.dataset.adminPage && user.role !== 'admin') {
    window.location.replace('dashboard.html');
    return null;
  }
  return user;
}

async function setupPortfolio() {
  if (!document.getElementById('risk-chart')) return;
  try {
    const [overview, risk, trends, eda] = await Promise.all([
      api('/api/portfolio/overview'), api('/api/portfolio/risk-distribution'),
      api('/api/portfolio/application-trends'), api('/api/portfolio/eda'),
    ]);
    document.getElementById('metric-applications').textContent = overview.total_loan_applications;
    document.getElementById('metric-total').textContent = money(overview.total_requested_loan_amount);
    document.getElementById('metric-average').textContent = money(overview.average_requested_loan_amount);
    document.getElementById('metric-review').textContent = overview.applications_under_review;
    document.getElementById('metric-low').textContent = overview.low_risk_applications;
    document.getElementById('metric-medium').textContent = overview.medium_risk_applications;
    document.getElementById('metric-high').textContent = overview.high_risk_applications;
    document.getElementById('metric-high-share').textContent = `${eda.insights.high_risk_percentage}%`;
    document.getElementById('insight-mean-loan').textContent = money(eda.insights.mean_loan_amount);
    document.getElementById('insight-median-loan').textContent = money(eda.insights.median_loan_amount);
    document.getElementById('insight-mean-income').textContent = money(eda.insights.mean_annual_income);
    document.getElementById('insight-credit').textContent = eda.insights.average_credit_score;
    document.getElementById('insight-growth').textContent = eda.insights.application_growth_percent_latest_month_over_previous === null ? 'Not available' : `${eda.insights.application_growth_percent_latest_month_over_previous}%`;
    document.getElementById('analytics-updated').textContent = `Updated ${new Intl.DateTimeFormat([], { dateStyle: 'medium', timeStyle: 'short' }).format(new Date())}`;
    if (typeof Chart === 'undefined') throw new Error('Chart library could not load. Check your internet connection and refresh this page.');
    drawPortfolioCharts(risk.distribution, trends.trends, eda);
  } catch (error) { messageFor(document.getElementById('analytics-error'), error.message); }
}

function drawPortfolioCharts(risk, trends, eda) {
  const colors = ['#37b6a5', '#e9b44c', '#d96c6c', '#4a8ca8', '#8b7cc1', '#5b9a77'];
  const chart = (id, type, labels, datasets, extra = {}) => new Chart(document.getElementById(id), {
    type, data: { labels, datasets: datasets.map((set, index) => ({
      backgroundColor: type === 'line' ? 'rgba(55,182,165,.15)' : colors[index % colors.length],
      borderColor: type === 'line' ? '#248e83' : '#fff', borderWidth: type === 'line' ? 2 : 1,
      borderRadius: type === 'bar' ? 5 : 0, tension: .32, ...set,
    })) }, options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: datasets.length > 1, position: 'bottom', labels: { boxWidth: 9, font: { size: 9 } } } }, scales: type === 'doughnut' ? {} : { y: { beginAtZero: true, grid: { color: '#eef2f4' }, ticks: { font: { size: 9 } } }, x: { grid: { display: false }, ticks: { font: { size: 9 } } } }, ...extra },
  });
  chart('risk-chart', 'doughnut', risk.map((item) => item.category), [{ data: risk.map((item) => item.count) }], { plugins: { legend: { display: true, position: 'bottom' } }, cutout: '63%' });
  chart('status-chart', 'bar', eda.application_status_distribution.map((item) => item.category), [{ label: 'Applications', data: eda.application_status_distribution.map((item) => item.count) }]);
  chart('amount-chart', 'bar', eda.requested_amount_distribution.map((item) => item.label), [{ label: 'Applications', data: eda.requested_amount_distribution.map((item) => item.count) }]);
  chart('credit-chart', 'bar', eda.credit_score_distribution.map((item) => item.label), [{ label: 'Customers', data: eda.credit_score_distribution.map((item) => item.count) }]);
  chart('employment-chart', 'doughnut', eda.employment_type_distribution.map((item) => item.category), [{ data: eda.employment_type_distribution.map((item) => item.count) }], { plugins: { legend: { display: true, position: 'bottom' } }, cutout: '63%' });
  chart('trend-chart', 'line', trends.map((item) => item.month), [{ label: 'Applications', data: trends.map((item) => item.count), fill: true }]);
  chart('income-chart', 'bar', eda.income_band_vs_requested_amount.map((item) => item.income_band), [{ label: 'Average requested amount (₹)', data: eda.income_band_vs_requested_amount.map((item) => item.average_requested_amount) }]);
  const categories = ['Low', 'Medium', 'High'];
  chart('risk-credit-chart', 'bar', eda.risk_category_vs_credit_score.map((item) => item.credit_score_band), categories.map((category) => ({ label: category, data: eda.risk_category_vs_credit_score.map((item) => item.counts[category]), backgroundColor: category === 'Low' ? '#37b6a5' : category === 'Medium' ? '#e9b44c' : '#d96c6c' })), { scales: { x: { stacked: true, grid: { display: false }, ticks: { font: { size: 9 } } }, y: { stacked: true, beginAtZero: true, grid: { color: '#eef2f4' }, ticks: { font: { size: 9 } } } } });
}

function setupChat() {
  const chatMessages = document.getElementById('chat-messages');
  const chatForm = document.getElementById('chat-form');
  if (!chatMessages || !chatForm) return;
  const messageInput = document.getElementById('message-input');
  const clearChatButton = document.getElementById('clear-chat');
  const initialWelcome = chatMessages.innerHTML;
  let pending = false;
  let sessionId = window.crypto?.randomUUID?.() || `session-${Date.now()}`;
  const scroll = () => { chatMessages.scrollTop = chatMessages.scrollHeight; };
  function renderMarkdown(md) {
    if (!md) return '';
    let html = escapeHTML(md);
    html = html.replace(/^### (.*$)/gim, '<h4 style="margin: 8px 0 4px; color: var(--navy); font-size: 13px;">$1</h4>');
    html = html.replace(/^## (.*$)/gim, '<h3 style="margin: 10px 0 4px; color: var(--navy); font-size: 14px;">$1</h3>');
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
    html = html.replace(/^[\*\-]\s+(.*)$/gim, '<span style="display:block; padding-left: 8px;">• $1</span>');
    html = html.replace(/\n\n/g, '<br><br>').replace(/\n/g, '<br>');
    return html;
  }
  function display(text, sender) {
    const row = document.createElement('div'); row.className = `message-row ${sender === 'user' ? 'user-row' : 'bot-row'}`;
    const avatar = document.createElement('div'); avatar.className = `avatar ${sender === 'user' ? 'user-avatar' : 'bot-avatar'}`; avatar.textContent = sender === 'user' ? 'Y' : 'S';
    const group = document.createElement('div'); group.className = 'message-group';
    const label = document.createElement('span'); label.className = 'sender-label'; label.textContent = sender === 'user' ? 'You' : 'SmartBank Assistant';
    const bubble = document.createElement('div'); bubble.className = `message-bubble ${sender === 'user' ? 'user-bubble' : 'bot-bubble'}`;
    if (sender === 'user') {
      const paragraph = document.createElement('p'); paragraph.textContent = text; bubble.appendChild(paragraph);
    } else {
      bubble.innerHTML = renderMarkdown(text);
    }
    group.append(label, bubble); row.append(avatar, group); chatMessages.appendChild(row); scroll();
  }
  chatForm.addEventListener('submit', async (event) => {
    event.preventDefault(); const text = messageInput.value.trim(); if (!text || pending) return;
    pending = true; display(text, 'user'); messageInput.value = '';
    const loading = document.createElement('div'); loading.className = 'message-row bot-row'; loading.textContent = 'SmartBank Assistant is typing…'; chatMessages.appendChild(loading);
    const waitNotice = window.setTimeout(() => { loading.textContent = 'Connecting to Gemini; local help will appear if it times out…'; }, 2500);
    try { const data = await api('/api/chat', { method: 'POST', body: JSON.stringify({ message: text, session_id: sessionId }) }); loading.remove(); display(data.bot_response, 'bot'); }
    catch (error) { loading.remove(); display(error.message, 'bot'); }
    finally { window.clearTimeout(waitNotice); pending = false; messageInput.focus(); }
  });
  clearChatButton?.addEventListener('click', async () => {
    const old = sessionId; sessionId = window.crypto?.randomUUID?.() || `session-${Date.now()}`;
    try { await api(`/api/chat/session/${encodeURIComponent(old)}`, { method: 'DELETE' }); } catch (_) { /* Nothing to clear on server. */ }
    chatMessages.innerHTML = initialWelcome; scroll(); messageInput.focus();
  });
  const docUploadInput = document.getElementById('doc-upload-input');
  const toolUploadBtn = document.getElementById('tool-upload-doc');
  const uploadedDocBar = document.getElementById('uploaded-doc-bar');
  const uploadedDocLabel = document.getElementById('uploaded-doc-label');
  const clearDocBtn = document.getElementById('clear-uploaded-doc');

  const headerRagBtn = document.getElementById('btn-header-rag');
  headerRagBtn?.addEventListener('click', () => docUploadInput?.click());
  toolUploadBtn?.addEventListener('click', () => docUploadInput?.click());

  // Check active indexed user documents on load
  (async () => {
    try {
      const { documents } = await api('/api/rag/documents');
      if (documents && documents.length > 0) {
        const latest = documents[documents.length - 1];
        if (uploadedDocBar && uploadedDocLabel) {
          uploadedDocBar.hidden = false;
          uploadedDocLabel.textContent = latest.filename;
        }
      }
    } catch (_) {}
  })();

  docUploadInput?.addEventListener('change', async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const loading = document.createElement('div');
    loading.className = 'message-row bot-row';
    loading.textContent = `Processing and indexing '${file.name}' into RAG knowledge base…`;
    chatMessages.appendChild(loading);
    scroll();

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await api('/api/rag/upload', {
        method: 'POST',
        body: formData,
      });
      loading.remove();
      if (uploadedDocBar && uploadedDocLabel) {
        uploadedDocBar.hidden = false;
        uploadedDocLabel.textContent = res.filename;
      }
      display(`📄 Successfully uploaded and indexed **${res.filename}** (${res.chunks_count} sections) into your RAG assistant. You can now ask any questions about this document!`, 'bot');
    } catch (error) {
      loading.remove();
      display(`❌ Document upload failed: ${error.message}`, 'bot');
    } finally {
      docUploadInput.value = '';
    }
  });

  clearDocBtn?.addEventListener('click', async () => {
    const currentName = uploadedDocLabel?.textContent;
    if (uploadedDocBar) uploadedDocBar.hidden = true;
    if (currentName) {
      try {
        await api(`/api/rag/documents/${encodeURIComponent(currentName)}`, { method: 'DELETE' });
        display(`Document **${currentName}** has been removed from active RAG indexing.`, 'bot');
      } catch (_) {}
    }
  });

  document.querySelectorAll('.suggestion-chip').forEach((button) => button.addEventListener('click', () => { messageInput.value = button.textContent; chatForm.requestSubmit(); }));
  document.querySelectorAll('[data-open-dialog]').forEach((button) => button.addEventListener('click', () => document.getElementById(button.dataset.openDialog)?.showModal()));
  document.querySelectorAll('[data-close-dialog]').forEach((button) => button.addEventListener('click', () => button.closest('dialog')?.close()));

  const inlineBtnFraud = document.getElementById('inline-btn-fraud') || document.getElementById('btn-agent-fraud');
  if (inlineBtnFraud) {
    inlineBtnFraud.addEventListener('click', () => {
      messageInput.value = 'Detect fraud in my account';
      chatForm.requestSubmit();
    });
  }

  const inlineBtnLoan = document.getElementById('inline-btn-loan') || document.getElementById('btn-agent-loan');
  if (inlineBtnLoan) {
    inlineBtnLoan.addEventListener('click', () => {
      messageInput.value = 'Check my loan eligibility';
      chatForm.requestSubmit();
    });
  }

  const dialogAutofillLoan = document.getElementById('dialog-autofill-loan');
  if (dialogAutofillLoan) {
    dialogAutofillLoan.addEventListener('click', async () => {
      try {
        dialogAutofillLoan.disabled = true;
        dialogAutofillLoan.textContent = 'Loading…';
        const profile = await api('/api/agent/loan-profile');
        const form = document.getElementById('loan-dialog')?.querySelector('form') || document.getElementById('loan-form');
        if (form && profile) {
          for (const [key, val] of Object.entries(profile)) {
            if (form.elements[key]) form.elements[key].value = val;
          }
        }
        dialogAutofillLoan.textContent = '✓ Filled';
        window.setTimeout(() => { dialogAutofillLoan.textContent = '⚡ Auto-fill profile'; dialogAutofillLoan.disabled = false; }, 1800);
      } catch (_) {
        dialogAutofillLoan.textContent = 'Error';
        dialogAutofillLoan.disabled = false;
      }
    });
  }
}

function showPredictionResult(container, kind, result) {
  if (!container) return;
  container.replaceChildren();
  const card = document.createElement('section');
  card.className = 'result-card';

  if (kind === 'loan') {
    const isEligible = result.prediction === 'Eligible';
    const prob = Number(result.predicted_probability || 0);
    card.innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; flex-wrap: wrap; gap: 8px;">
        <h3 style="margin: 0; font-size: 16px;">
          Model Decision: <span class="risk-badge ${isEligible ? 'low' : 'high'}" style="font-size: 13px; padding: 4px 10px;">${escapeHTML(result.prediction)}</span>
        </h3>
        <strong style="color: ${isEligible ? '#155c5a' : '#a63434'}; font-size: 14px;">${(prob * 100).toFixed(1)}% Approval Probability</strong>
      </div>
      <div style="background: #e9f5f2; border-radius: 8px; height: 10px; width: 100%; overflow: hidden; margin-bottom: 12px;">
        <div style="background: ${isEligible ? '#24795f' : '#dc3545'}; height: 100%; width: ${(prob * 100).toFixed(1)}%;"></div>
      </div>
      ${result.key_influencing_features && result.key_influencing_features.length ? `
        <div style="margin: 10px 0;">
          <strong style="font-size: 12px; color: #1b4157;">Key Model Features:</strong>
          <ul style="margin: 6px 0 10px; padding-left: 18px; font-size: 12px; color: #526878;">
            ${result.key_influencing_features.map(f => `<li><strong>${escapeHTML(f.feature)}</strong>: importance ${f.importance}</li>`).join('')}
          </ul>
        </div>
      ` : ''}
      <p style="margin: 8px 0; color: #526878; font-size: 12px; line-height: 1.5;">${escapeHTML(result.explanation || '')}</p>
      <div style="display: flex; gap: 10px; margin-top: 15px; flex-wrap: wrap;">
        <a class="primary-button link-button" href="apply-loan.html" style="font-size: 12px; padding: 7px 14px;">Apply for this loan →</a>
        <a class="secondary-button" href="index.html" style="font-size: 12px; padding: 7px 14px;">Consult AI Assistant</a>
      </div>
      <p class="muted" style="margin-top: 10px; font-size: 10px;">Experimental estimate from synthetic training data; not a formal credit decision.</p>
    `;
  } else {
    const risk = result.risk_level || 'Low';
    const prob = Number(result.fraud_probability || 0);
    const riskClass = risk.toLowerCase();
    card.innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; flex-wrap: wrap; gap: 8px;">
        <h3 style="margin: 0; font-size: 16px;">
          Risk Analysis: <span class="risk-badge ${riskClass}" style="font-size: 13px; padding: 4px 10px;">${escapeHTML(risk)} Risk (${escapeHTML(result.prediction)})</span>
        </h3>
        <strong style="color: ${risk === 'High' ? '#a63434' : risk === 'Medium' ? '#9a6b20' : '#24795f'}; font-size: 14px;">
          ${(prob * 100).toFixed(1)}% Fraud Probability
        </strong>
      </div>
      <div style="background: #f1f5f7; border-radius: 8px; height: 10px; width: 100%; overflow: hidden; margin-bottom: 12px;">
        <div style="background: ${risk === 'High' ? '#dc3545' : risk === 'Medium' ? '#f0ad4e' : '#28a745'}; height: 100%; width: ${Math.max(5, (prob * 100)).toFixed(1)}%;"></div>
      </div>
      ${result.main_suspicious_characteristics && result.main_suspicious_characteristics.length ? `
        <div style="margin: 10px 0;">
          <strong style="font-size: 12px; color: #1b4157;">Risk Indicators Evaluated:</strong>
          <ul style="margin: 6px 0 10px; padding-left: 18px; font-size: 12px; color: #526878;">
            ${result.main_suspicious_characteristics.map(c => `<li>${escapeHTML(c)}</li>`).join('')}
          </ul>
        </div>
      ` : ''}
      <p style="margin: 8px 0; color: #526878; font-size: 12px; line-height: 1.5;">${escapeHTML(result.recommendation || '')}</p>
      <div style="display: flex; gap: 10px; margin-top: 15px; flex-wrap: wrap;">
        <a class="secondary-button" href="index.html" style="font-size: 12px; padding: 7px 14px;">Ask AI Assistant for Help</a>
        <a class="secondary-button" href="transactions.html" style="font-size: 12px; padding: 7px 14px;">View Ledger</a>
      </div>
      <p class="muted" style="margin-top: 10px; font-size: 10px;">Statistical risk scoring only; transactions are never blocked automatically by this demo.</p>
    `;
  }
  container.appendChild(card);
}

function setupPredictionForms() {
  for (const kind of ['loan', 'fraud']) {
    const form = document.getElementById(`${kind}-form`); if (!form) continue;
    form.addEventListener('submit', async (event) => {
      event.preventDefault(); const target = document.getElementById(`${kind}-result`); const button = form.querySelector('button[type="submit"]');
      target.textContent = 'Running estimate…'; button.disabled = true;
      try { showPredictionResult(target, kind, await api(`/api/${kind}/predict`, { method: 'POST', body: JSON.stringify(formPayload(form)) })); }
      catch (error) { target.textContent = error.message; }
      finally { button.disabled = false; }
    });
  }
}

async function setupLoanEligibilityPage() {
  const autofillBtn = document.getElementById('autofill-loan-btn');
  if (autofillBtn) {
    autofillBtn.addEventListener('click', async () => {
      try {
        autofillBtn.disabled = true;
        autofillBtn.textContent = 'Loading profile…';
        const profile = await api('/api/agent/loan-profile');
        if (document.getElementById('loan-age')) document.getElementById('loan-age').value = profile.age;
        if (document.getElementById('loan-income')) document.getElementById('loan-income').value = profile.annual_income;
        if (document.getElementById('loan-employment')) document.getElementById('loan-employment').value = profile.employment_type;
        if (document.getElementById('loan-experience')) document.getElementById('loan-experience').value = profile.employment_experience;
        if (document.getElementById('loan-credit')) document.getElementById('loan-credit').value = profile.credit_score;
        if (document.getElementById('loan-existing')) document.getElementById('loan-existing').value = profile.existing_loan_amount;
        if (document.getElementById('loan-debt')) document.getElementById('loan-debt').value = profile.monthly_debt;
        if (document.getElementById('loan-requested')) document.getElementById('loan-requested').value = profile.requested_loan_amount;
        autofillBtn.textContent = '✓ Profile Loaded';
        window.setTimeout(() => { autofillBtn.textContent = '⚡ Fill from my profile'; autofillBtn.disabled = false; }, 2000);
      } catch (e) {
        autofillBtn.textContent = 'Failed to load';
        autofillBtn.disabled = false;
      }
    });
  }
}

async function setupFraudDetectionPage() {
  const scanTable = document.getElementById('fraud-scan-table');
  const scanAllBtn = document.getElementById('scan-all-tx-btn');
  const statusBadge = document.getElementById('security-status-badge');

  async function loadAndScanTransactions() {
    if (!scanTable) return;
    try {
      scanTable.innerHTML = '<p class="muted">Scanning account ledger with AI fraud pipeline…</p>';
      const data = await api('/api/agent/scan-transactions-fraud');
      if (statusBadge) {
        statusBadge.textContent = data.account_security_status || 'Secure';
        statusBadge.className = `status-pill ${data.account_security_status === 'Secure' ? 'status-active' : 'status-review'}`;
      }
      if (!data.transactions || !data.transactions.length) {
        scanTable.innerHTML = '<p class="muted">No transactions available to scan.</p>';
        return;
      }
      const rows = data.transactions.slice(0, 15).map(t => {
        const risk = t.risk_level || 'Low';
        const riskClass = risk.toLowerCase();
        return `
          <tr>
            <td>${escapeHTML(shortDate(t.date))}</td>
            <td><strong>${escapeHTML(t.description)}</strong><br><small class="muted">${escapeHTML(t.channel || 'Online')}</small></td>
            <td>${escapeHTML(t.merchant_category || 'Other')}</td>
            <td>${money(t.amount)}</td>
            <td><span class="risk-badge ${riskClass}">${risk} Risk</span></td>
            <td>
              <button class="secondary-button tx-scan-btn" type="button" style="padding: 4px 8px; font-size: 11px;"
                data-amount="${t.amount}"
                data-hour="${(t.time || '12:00:00').split(':')[0]}"
                data-cat="${escapeHTML(t.merchant_category || 'Retail')}"
                data-channel="${escapeHTML(t.channel || 'Online')}">
                Test in Simulator
              </button>
            </td>
          </tr>
        `;
      }).join('');
      scanTable.innerHTML = `
        <table class="data-table">
          <thead>
            <tr>
              <th>Date</th>
              <th>Description</th>
              <th>Category</th>
              <th>Amount</th>
              <th>AI Risk Score</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      `;

      scanTable.querySelectorAll('.tx-scan-btn').forEach(btn => {
        btn.addEventListener('click', () => {
          if (document.getElementById('fraud-amount')) document.getElementById('fraud-amount').value = btn.dataset.amount;
          if (document.getElementById('fraud-hour')) document.getElementById('fraud-hour').value = btn.dataset.hour;
          if (document.getElementById('fraud-category')) document.getElementById('fraud-category').value = btn.dataset.cat;
          document.getElementById('fraud-form')?.scrollIntoView({ behavior: 'smooth' });
        });
      });
    } catch (e) {
      scanTable.innerHTML = `<p class="muted">Could not scan transactions: ${escapeHTML(e.message)}</p>`;
    }
  }

  if (scanTable) {
    await loadAndScanTransactions();
  }
  if (scanAllBtn) {
    scanAllBtn.addEventListener('click', async () => {
      scanAllBtn.disabled = true;
      scanAllBtn.textContent = 'Auditing ledger…';
      await loadAndScanTransactions();
      scanAllBtn.textContent = '✓ Scan complete';
      window.setTimeout(() => { scanAllBtn.textContent = '🤖 AI scan account transactions'; scanAllBtn.disabled = false; }, 2000);
    });
  }
}

(async function initialize() {
  setupAuthForms(); setupLogout(); setupChat(); setupPredictionForms(); setupLoanApplication();
  await setupLoanEligibilityPage(); await setupFraudDetectionPage();
  const user = await setupProtectedPage();
  if (user) { await setupCustomerPages(user); await setupApplicationDetails(); await setupPortfolio(); await setupBankingPages(); }
})();
