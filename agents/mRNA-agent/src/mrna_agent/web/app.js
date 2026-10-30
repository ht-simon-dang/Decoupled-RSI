'use strict';
const $ = id => document.getElementById(id);
const state = {input:'form', run:null, key:null, examples:[], runs:[], brief:{}, selected:null, polling:false, mode:'preview', request:0, viewer:null, detailKey:null, structure:'rna'};
const terminal = new Set(['complete','failed','blocked','paused','cancelled','budget_exhausted']);
const names = {created:'已创建',queued:'排队中',running:'运行中',planning:'解析与规划',planned:'规划完成',generating:'生成候选',generated:'生成完成',validating:'约束核验',predicting:'属性预测',predicted:'预测完成',evaluating:'完整构建体评价',selecting:'多目标选优',complete:'已完成',failed:'运行失败',blocked:'需要补充信息',paused:'已暂停',cancelled:'已取消',budget_exhausted:'预算耗尽',evolving:'流程审查中'};
const roles = {high_potential:'High-potential',control:'参考对照',paired_hypothesis:'配对假设',diversity:'多样性',exploration:'探索',measured_followup:'实测后续验证'};
const reasons = {no_high_potential_candidates:'没有候选通过高潜力筛选；未交付序列',needs_payload:'需要确定蛋白',recommendations_ready:'计算候选已形成',needs_experimental_data:'需要实验数据',insufficient_candidates:'合格候选不足',all_constraints_failed:'约束均未通过',budget_exhausted:'预算耗尽'};
function node(tag, text, cls) {const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;}
function showError(e) {notice(e.message || String(e));}
let noticeTimer;
function notice(text) {$('notice').textContent=text;$('notice').hidden=false;clearTimeout(noticeTimer);noticeTimer=setTimeout(()=>$('notice').hidden=true,7000);}
async function api(path, options={}) {const r=await fetch('/api'+path,{...options,headers:{'Content-Type':'application/json',...(options.headers||{})}});const body=await r.json();if(!r.ok)throw new Error(typeof body.detail==='string'?body.detail:JSON.stringify(body.detail||body));return body;}
const post = (path, body={}) => api(path,{method:'POST',body:JSON.stringify(body)});
const num = value => value == null ? '—' : Number(value).toFixed(3);
const metric = (e,k) => (e?.metrics||[]).find(m=>m.metric===k && m.status==='ok');
function setView(view) {document.querySelectorAll('.view').forEach(n=>n.hidden=n.id!=='view-'+view);document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('active',b.dataset.view===view));if(view==='models')loadModels().catch(showError);if(view==='evolution')loadEvolution().catch(showError);if(view==='feedback')loadHistory().catch(showError);}
document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>setView(b.dataset.view)));
function formBrief() {
  const b=structuredClone(state.brief||{});b.name=$('task-name').value.trim()||'mRNA design';b.protein=$('protein').value.replace(/\s/g,'')||null;b.mode=$('mode').value;
  b.context={...(b.context||{}),target_cell:$('cell').value.trim()||'unspecified'};
  b.molecule={...(b.molecule||{}),cap:$('cap').value,modification:$('modification').value,poly_a_length:Number($('poly-a').value)};
  b.budget={...(b.budget||{}),candidates:Number($('count').value),shortlist:Number($('shortlist').value),rounds:Number($('rounds').value)};b.seed=Number($('seed').value);
  const primary=$('objective').value;const old=(b.objectives||[]).find(o=>o.role==='primary');
  if(!old||old.metric!==primary)b.objectives=[{metric:primary,role:'primary',direction:primary==='chemical_degradation'?'minimize':'maximize',stage:['translation_efficiency','start_accessibility','ribosome_load','rna_stability','chemical_degradation'].includes(primary)?'computational':'experimental'}];
  const cds=$('cds').value.replace(/\s/g,'');if(cds)b.reference={utr5:$('utr5').value.replace(/\s/g,''),cds,utr3:$('utr3').value.replace(/\s/g,'')};else delete b.reference;
  return b;
}
function fillForm(b) {
  state.brief=structuredClone(b);$('task-name').value=b.name||'mRNA design';$('protein').value=b.protein||'';$('mode').value=b.mode||'full';$('cell').value=b.context?.target_cell||'unspecified';
  $('modification').value=b.molecule?.modification||'unspecified';$('cap').value=b.molecule?.cap||'unspecified';$('poly-a').value=b.molecule?.poly_a_length||0;$('seed').value=b.seed??42;
  $('count').value=b.budget?.candidates||8;$('shortlist').value=b.budget?.shortlist||4;$('rounds').value=b.budget?.rounds||1;$('objective').value=b.objectives?.find(o=>o.role==='primary')?.metric||'protein_auc';
  for(const r of ['utr5','cds','utr3'])$(r).value=b.reference?.[r]||'';
  $('brief-json').value=JSON.stringify(b,null,2);
}
document.querySelectorAll('[data-input]').forEach(b=>b.addEventListener('click',()=>{
  try {if(state.input==='form')state.brief=formBrief();if(state.input==='json')state.brief=JSON.parse($('brief-json').value);if(b.dataset.input==='json')$('brief-json').value=JSON.stringify(state.brief,null,2);if(b.dataset.input==='form')fillForm(state.brief);
    state.input=b.dataset.input;for(const v of ['form','instruction','json'])$('input-'+v).hidden=v!==state.input;document.querySelectorAll('[data-input]').forEach(x=>x.classList.toggle('active',x===b));
  } catch(e){showError(e);}
}));
$('preset').addEventListener('change',()=>{const p=state.examples.find(e=>e.id===$('preset').value);if(p)fillForm(p.brief);});
$('task-form').addEventListener('submit',async e=>{e.preventDefault();$('launch').disabled=true;try{const body=state.input==='instruction'?{instruction:$('instruction').value}:{brief:state.input==='json'?JSON.parse($('brief-json').value):formBrief()};const r=await post('/runs',body);await selectRun(r.id,'delivery');await loadHistory();notice('任务已提交，运行记录将实时更新。');}catch(e){showError(e);}finally{$('launch').disabled=false;}});
function clearDetail(){state.viewer?.dispose();state.viewer=null;state.detailKey=null;$('candidate-detail').hidden=true;$('candidate-detail').replaceChildren();}
function route(push=false){const p=new URLSearchParams({run:state.key,mode:state.mode,structure:state.structure});if(state.selected)p.set('candidate',state.selected);const url='#'+p.toString();if(location.hash!==url)history[push?'pushState':'replaceState'](null,'',url);}
async function selectRun(key,mode='preview',candidate=null,push=true,scroll=true) {
  state.request++;state.key=key;state.mode=mode;state.selected=candidate;state.run=null;clearDetail();
  $('results-panel').hidden=true;for(const action of ['pause','resume','reevaluate'])$(action).disabled=true;$('candidates').replaceChildren();$('run-title').textContent='正在打开任务…';$('run-meta').textContent=key;
  route(push);setView('design');await refreshRun();
  if(scroll&&state.key===key)$('run-title').scrollIntoView({block:'start',behavior:'instant'});
}
async function refreshRun() {
  if(!state.key)return;const key=state.key,mode=state.mode,request=++state.request;
  try{const run=await api('/runs/'+encodeURIComponent(key)+(mode==='preview'?'/preview':''));if(request!==state.request)return;state.run=run;renderRun(run);}
  catch(e){if(request!==state.request)return;$('run-title').textContent='任务无法打开';throw e;}
}
function selectCandidate(id,scroll=false){const c=state.run?.payload.candidates.find(c=>c.id===id);if(!c)return;state.selected=id;route();renderDetail(c,state.run.payload.evaluations.find(e=>e.construct_id===id),state.run);$('candidates').querySelectorAll('tr').forEach(n=>n.classList.toggle('selected',n.dataset.id===id));if(scroll)$('candidate-detail').scrollIntoView({block:'start',behavior:'instant'});}
window.addEventListener('popstate',()=>{const p=new URLSearchParams(location.hash.slice(1));state.structure=p.get('structure')==='protein'?'protein':'rna';const key=p.get('run')||state.runs[0]?.id;if(key)selectRun(key,p.get('mode')==='delivery'?'delivery':'preview',p.get('candidate'),false).catch(showError);});
for(const mode of ['preview','delivery'])$('mode-'+mode).addEventListener('click',()=>{if(state.key)selectRun(state.key,mode,null,true,false).catch(showError);});
$('run-select').addEventListener('change',()=>{if($('run-select').value)selectRun($('run-select').value).catch(showError);});
$('copy-brief').addEventListener('click',()=>{if(state.run){fillForm(state.run.brief);$('task-form').scrollIntoView({block:'start'});notice('历史任务输入已载入。编辑后可作为新任务启动。');}});
function renderRun(run) {
  const p=run.payload||{};const es=p.evaluations||[];const cs=p.candidates||[];const port=p.portfolio||{};
  for(const action of ['pause','resume','reevaluate'])$(action).disabled=false;
  const preview=Boolean(run.preview);if(!cs.some(c=>c.id===state.selected))state.selected=cs[0]?.id||null;route();
  $('run-meta').textContent=run.id+' · '+new Date(run.created*1000).toLocaleString();$('run-select').value=run.id;
  for(const mode of ['preview','delivery'])$('mode-'+mode).classList.toggle('active',state.mode===mode);
  $('preview-notice').hidden=!preview;$('preview-notice').textContent=(run.preview?.legacy?'旧版任务，未通过当前强制评价协议。':'')+'只读历史预览：包括参考对照、未推荐及已推荐记录；打开预览不会改变评价或筛选结果。';
  $('run-brief').hidden=false;$('run-brief-json').textContent=JSON.stringify(run.brief,null,2);
  $('results-title').textContent=preview?'历史构建体预览 · '+cs.length+' 条记录':'High-potential 推荐';
  $('results-empty').hidden=cs.length>0;$('results-empty').textContent=preview?'本任务没有已保存的构建体。可查看运行轨迹和任务输入。':'本任务没有通过当前筛选的推荐；点击“历史预览”查看原始评价记录。';
  document.querySelectorAll('.history-item').forEach(n=>n.classList.toggle('active',n.dataset.id===run.id));
  $('run-title').textContent=run.brief.name||'Natural language task';$('run-state').textContent=names[run.state]||run.state;
  $('stat-count').textContent=p.delivery?.generated_count||0;$('stat-valid').textContent=p.delivery?.evaluated_count||0;$('stat-frontier').textContent=(port.frontier||[]).length;$('stat-measured').textContent=es.filter(e=>e.experimental_success).length;
  $('pause').hidden=terminal.has(run.state);$('resume').hidden=!['paused','failed','blocked'].includes(run.state);$('reevaluate').hidden=!p.delivery?.generated_count||!terminal.has(run.state);
  const stages=['目标与规划','区域生成','约束核验','完整评价','候选组合','证据报告'];const indices={planning:0,planned:0,generating:1,generated:1,validating:2,predicting:3,predicted:3,evaluating:3,selecting:4,complete:5};const current=indices[run.state]??-1;
  $('pipeline').replaceChildren(...stages.map((s,i)=>node('div',String(i+1).padStart(2,'0')+' '+s,i<current?'done':i===current?'current':'')));
  $('plan').textContent=p.error||p.plan?.rationale||p.report?.recommendation||'正在读取目标与任务约束…';
  const events=(run.events||[]).filter(e=>!['state','created'].includes(e.kind)).slice(-24);
  $('timeline').replaceChildren(...events.map(e=>{const r=node('div',undefined,'event');r.append(node('time',new Date(e.time*1000).toLocaleTimeString()),node('span',e.kind,'event-kind'));let detail='';const d=e.payload;
    detail='执行记录 · 原始模型输入与未筛选序列保留在本地审计日志';r.append(node('p',detail));return r;}));
  if(!events.length)$('timeline').append(node('div',p.report&&typeof p.report==='string'?p.report:'任务已记录。等待下一阶段。','empty-note'));
  $('results-panel').hidden=!terminal.has(run.state)&&!cs.length;$('stop-reason').textContent=reasons[p.stop_reason]||p.stop_reason||'';
  for(const f of ['json','csv','fasta'])$('export-'+f).href='/api/runs/'+run.id+'/export/'+f;
  $('candidates').replaceChildren(...cs.map(c=>{const e=es.find(e=>e.construct_id===c.id);const role=(port.selected||[]).find(s=>s.id===c.id)?.role;const tr=node('tr');tr.dataset.id=c.id;tr.tabIndex=0;tr.setAttribute('aria-label','查看构建体 '+c.id.slice(0,10));tr.classList.toggle('selected',state.selected===c.id);
    const id=node('td',c.id.slice(0,10));id.append(node('small',preview?({high_potential:'High-potential · 已推荐',control:'参考对照 · 非推荐',legacy_unverified:'旧版记录 · 未经当前验证',not_evaluated:'尚未评价 · 非推荐',not_delivered:'未入选推荐'}[c.preview?.status]||'历史记录'):roles[role]||'候选'));tr.append(id,node('td',c.sequence.length),node('td',num(metric(e,'translation_efficiency')?.value)),node('td',num(metric(e,'start_accessibility')?.value)),node('td',e?.status==='rejected'?'未通过':e?.status==='pending_constraints'?'待验证':e?'计算通过':'待评价',e?.status==='rejected'?'fail':e?.status==='pending_constraints'?'pending':'pass'),node('td',(port.frontier||[]).includes(c.id)?'●':'—'));
    const open=()=>selectCandidate(c.id,true);tr.addEventListener('click',open);tr.addEventListener('keydown',e=>{if(e.key==='Enter')open();});return tr;}));
  $('limitations').textContent=JSON.stringify({delivery:p.delivery,filter_policy:port.filter_policy,filter_results:port.filter_results,report:p.report},null,2);
  plot(cs,es,port);
  if(state.selected&&cs.some(c=>c.id===state.selected)){const c=cs.find(c=>c.id===state.selected);renderDetail(c,es.find(e=>e.construct_id===c.id),run);}else clearDetail();
  if(p.operation==='evolution'&&p.result)loadEvolution().catch(showError);
}
function renderDetail(c,e,run) {
  const detailKey=[run.id,c.id,e?.evaluation_id||e?.status,state.mode].join(':');if(state.detailKey===detailKey)return;clearDetail();state.detailKey=detailKey;
  const box=$('candidate-detail');box.hidden=false;box.replaceChildren(node('h3','构建体证据卡 · '+c.id.slice(0,14)));
  box.append(node('p',`${c.molecule.cap} · ${c.molecule.modification} · poly(A) ${c.molecule.poly_a_length} nt · ${c.protein_version}`,'muted'));
  if(c.preview&&!c.preview.recommended)box.append(node('p',c.preview.status==='control'?'参考对照 · 不属于 high-potential 推荐':'历史证据预览 · 该构建体不属于当前 high-potential 推荐','preview-notice'));
  const viewer=node('section');box.append(viewer);state.viewer=new StructureViewer(viewer,run.id,c.id,state.structure,kind=>{state.structure=kind;route();});
  const seq=node('div',undefined,'sequence');for(const r of ['utr5','cds','utr3']){seq.append(node('strong',r.toUpperCase()+' ['+c.boundaries[r].join(',')+') '));seq.append(node('span',c.regions[r]+' ',r==='cds'?'cds':'utr'));}seq.append(node('span','A'.repeat(c.molecule.poly_a_length),'tail'));box.append(seq);
  const table=node('table');const head=node('tr');for(const x of ['指标','数值 / 单位','状态 / 证据','来源','独立于优化'])head.append(node('th',x));table.append(head);
  for(const m of e?.metrics||[]){const tr=node('tr');tr.append(node('td',m.metric),node('td',m.status==='ok'?num(m.value)+' '+m.unit:'—'),node('td',m.status+' / '+m.evidence),node('td',m.source),node('td',m.independent?'是':'否'));tr.title=m.reason||m.scope;table.append(tr);}const wrap=node('div',undefined,'table-wrap');wrap.append(table);box.append(wrap);
  const details=node('details');details.append(node('summary','完整来源、固定约束和结构证据'),node('pre',JSON.stringify({construct:c,evaluation:e},null,2)));box.append(details);
}
function svgNode(tag,attrs,text){const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text)e.textContent=text;return e;}
function plot(cs,es,p) {
  const svg=$('tradeoff');svg.replaceChildren();const pts=es.map(e=>({id:e.construct_id,x:metric(e,'start_accessibility')?.value,y:metric(e,'translation_efficiency')?.value})).filter(r=>r.x!=null&&r.y!=null);
  $('chart-wrap').hidden=pts.length<2;if(pts.length<2)return;
  const xmin=Math.min(...pts.map(p=>p.x)),xmax=Math.max(...pts.map(p=>p.x));const ymin=Math.min(...pts.map(p=>p.y)),ymax=Math.max(...pts.map(p=>p.y));const sx=x=>65+(x-xmin)/(xmax-xmin||1)*570;const sy=y=>185-(y-ymin)/(ymax-ymin||1)*150;
  for(let i=0;i<5;i++){const y=185-i*37.5;svg.append(svgNode('line',{x1:65,x2:650,y1:y,y2:y}));svg.append(svgNode('text',{x:5,y:y+3},num(ymin+(ymax-ymin)*i/4)));}
  svg.append(svgNode('text',{x:250,y:220},'Start accessibility · mean unpaired probability'),svgNode('text',{x:65,y:17},'Predicted normalized TE'));
  for(const pt of pts){const circle=svgNode('circle',{cx:sx(pt.x),cy:sy(pt.y),r:6,class:(p.frontier||[]).includes(pt.id)?'frontier':''});circle.append(svgNode('title',{},pt.id.slice(0,10)+' · '+num(pt.x)+' / '+num(pt.y)));circle.addEventListener('click',()=>selectCandidate(pt.id,true));svg.append(circle);}
  svg.append(svgNode('text',{x:65,y:202},num(xmin)),svgNode('text',{x:620,y:202},num(xmax)));
}
async function loadHistory() {
  state.runs=await api('/runs');$('history').replaceChildren(...state.runs.map(r=>{const row=node('div',undefined,'history-item');row.dataset.id=r.id;row.classList.toggle('active',r.id===state.key);const b=node('button',r.name);b.append(node('small',r.id.slice(0,12)+' · '+new Date(r.created*1000).toLocaleString()));b.addEventListener('click',()=>selectRun(r.id).catch(showError));row.append(b,node('span',r.count+' 构建体','muted'),node('span',names[r.state]||r.state,'badge'));return row;}));if(!state.runs.length)$('history').append(node('div','暂无任务记录','empty-note'));
  const current=state.key;$('run-select').replaceChildren(node('option','选择过往任务…'));$('run-select').firstChild.value='';for(const r of state.runs){const option=node('option',r.name+' · '+r.id.slice(0,8)+' · '+r.count+' 构建体');option.value=r.id;$('run-select').append(option);}if(current)$('run-select').value=current;
  const previous=$('feedback-run').value;$('feedback-run').replaceChildren(...state.runs.filter(r=>r.count>0).map(r=>{const o=node('option',r.name+' · '+r.id.slice(0,8));o.value=r.id;return o;}));if(previous)$('feedback-run').value=previous;else if(state.key)$('feedback-run').value=state.key;
}
$('refresh').addEventListener('click',()=>Promise.all([refreshRun(),loadHistory()]).catch(showError));
for(const op of ['pause','resume'])$(op).addEventListener('click',async()=>{try{await post('/runs/'+state.key+'/'+op);await refreshRun();await loadHistory();}catch(e){showError(e);}});
$('reevaluate').addEventListener('click',async()=>{try{const r=await post('/runs/'+state.key+'/reevaluate');await selectRun(r.id,'delivery');await loadHistory();}catch(e){showError(e);}});
async function loadModels() {const d=await api('/capabilities');$('disk-usage').textContent=(d.data_bytes/1e9).toFixed(2)+' / '+(d.quota_bytes/1e9)+' GB';$('model-cards').replaceChildren(...d.models.map(m=>{const c=node('article',undefined,'model-card');const h=node('div',undefined,'model-head');h.append(node('h3',m.id),node('span',m.enabled?'已启用':'缺少官方资源','badge'));c.append(h,node('small',m.role+' · '+(m.regions.join(' / ')||'orchestration')),node('p',m.scope),node('small',m.license));const a=node('a','作者资源 ↗');if(m.source.startsWith('https://')){a.href=m.source;a.target='_blank';a.rel='noopener noreferrer';c.append(a);}return c;}));}
async function loadEvolution() {const d=await api('/evolution');$('active-workflow').textContent=d.active_workflow;const box=$('evolution-history');box.replaceChildren();if(!d.cycles.length)box.append(node('p','尚无审查记录。候选流程须通过开发、审计和最终留出任务后才能晋升。','muted'));for(const c of d.cycles){box.append(node('h3',c.baseline+' → '+c.candidate+' · '+(c.decision.promote?'已晋升':'保留原流程')),node('p',c.decision.reason,'muted'),node('pre',JSON.stringify(c,null,2)));}const detail=node('details');detail.append(node('summary','固定协议与哈希'),node('pre',JSON.stringify({protocol:d.protocol,hash:d.protocol_hash},null,2)));box.append(detail);}
$('evolve').addEventListener('click',async()=>{try{const r=await post('/evolution');await selectRun(r.id,'delivery');await loadHistory();notice('流程对照审查已排队。');}catch(e){showError(e);}});
$('rollback').addEventListener('click',async()=>{try{await post('/evolution/rollback');await loadEvolution();notice('已回退至 v1，历史记录保留。');}catch(e){showError(e);}});
$('experiment-template').addEventListener('click',async()=>{try{const key=$('feedback-run').value;if(!key)throw Error('请先完成一个设计任务。');const r=await api('/runs/'+key);const baseline=r.payload.report?.baseline;if(!r.payload.candidates.length)throw Error('本任务没有通过筛选的候选，无法生成推荐实验模板。');const records=r.payload.candidates.slice(0,3).map(c=>({construct_id:c.id,context:r.brief.context,metric:'protein_auc',value:null,unit:'a.u.·h',time_hours:48,batch:'batch-1',replicate:'1',control_id:baseline,study:'study-1',assay:'reporter',quality:'missing'}));$('experiment-json').value=JSON.stringify(records,null,2);notice('已生成空值模板；填入真实测量值后将 quality 改为 valid。');}catch(e){showError(e);}});
function renderFeedback(d){const box=$('feedback-results');box.replaceChildren(node('p',`有效实验记录 ${d.valid_experimental_records} / 接收记录 ${d.records}`));for(const x of d.next_batch){box.append(node('h3',x.id.slice(0,12)+' · '+(roles[x.role]||x.role)),node('p',x.reason));}box.append(node('pre',JSON.stringify(d,null,2)));}
$('analyze-feedback').addEventListener('click',async()=>{try{const k=$('feedback-run').value;if(!k)throw Error('请选择有候选的任务。');renderFeedback(await post('/runs/'+k+'/feedback'));}catch(e){showError(e);}});
$('import-experiments').addEventListener('click',async()=>{try{const records=JSON.parse($('experiment-json').value);const r=await post('/experiments',{records});notice('已接收 '+r.received+' 条记录。');renderFeedback(await post('/runs/'+$('feedback-run').value+'/feedback'));}catch(e){showError(e);}});
async function init(){const h=await api('/health');$('health').textContent='LOCAL · '+h.version;state.examples=await api('/examples');$('preset').replaceChildren(node('option','选择一个示例任务'));$('preset').firstChild.value='';for(const e of state.examples){const o=node('option',e.brief.name);o.value=e.id;$('preset').append(o);}const first=state.examples.find(e=>e.id==='hbb')||state.examples[0];if(first)fillForm(first.brief);await loadHistory();const params=new URLSearchParams(location.hash.slice(1));state.structure=params.get('structure')==='protein'?'protein':'rna';const key=params.get('run')||state.runs[0]?.id;if(key)await selectRun(key,params.get('mode')==='delivery'?'delivery':'preview',params.get('candidate'),false,Boolean(params.get('run')));}
init().catch(e=>{$('health').textContent='连接失败';showError(e);});
setInterval(async()=>{if(state.polling||!state.key||terminal.has(state.run?.state))return;state.polling=true;try{await refreshRun();if(terminal.has(state.run?.state))await loadHistory();}catch(e){showError(e);}finally{state.polling=false;}},2000);
