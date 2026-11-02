"use strict";
const $ = id => document.getElementById(id);
const state = {brief:null, mode:"structured", run:null, selected:null, timer:null, loading:false};
const viewer = new window.AntibodyStructure($("structure"));
const terminal = new Set(["complete","blocked","failed","cancelled","budget_exhausted","interrupted"]);
const pretty = value => JSON.stringify(value,null,2);
function node(tag,text,className) {const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(className)n.className=className;return n;}
function showError(error) {$("notice").textContent=error.message||String(error);$("notice").hidden=false;clearTimeout(showError.timer);showError.timer=setTimeout(()=>{$("notice").hidden=true;},9000);}
async function api(path,body) {
  const response=await fetch(path,body===undefined?{}:{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  if(!response.ok){let message;try{const data=await response.json();message=typeof data.detail==="string"?data.detail:pretty(data.detail);}catch{message=response.statusText;}throw new Error(message);}
  return response.json();
}
function action(id,fn){$(id).addEventListener("click",async()=>{const b=$(id);if(b.disabled)return;b.disabled=true;try{await fn();}catch(e){showError(e);}finally{b.disabled=false;if(id==="submit-review"&&!state.selected)b.disabled=true;}});}
function requireRun(){if(!state.run)throw new Error("请先选择或启动一个任务。");return state.run.id;}
function showView(name){document.querySelectorAll(".view").forEach(el=>{el.hidden=el.id!=="view-"+name;});document.querySelectorAll(".tab").forEach(el=>el.classList.toggle("active",el.dataset.view===name));}
document.querySelectorAll(".tab").forEach(el=>el.addEventListener("click",()=>showView(el.dataset.view)));
function setMode(mode){state.mode=mode;$("natural-form").hidden=mode!=="natural";$("structured-form").hidden=mode!=="structured";$("natural-mode").classList.toggle("active",mode==="natural");$("structured-mode").classList.toggle("active",mode==="structured");}
action("natural-mode",()=>setMode("natural"));action("structured-mode",()=>setMode("structured"));
async function loadTemplate(){state.brief=await api("/api/templates/"+$("template").value);const b=state.brief;for(const k of ["name","entry","route","direction"])$(k).value=b[k];$("disease").value=b.disease==="unspecified"?"":b.disease;$("format").value=b.product.format;$("generator").value=b.generators[0];$("count").value=b.budget.candidates;$("brief-json").value=pretty(b);}
$("template").addEventListener("change",()=>loadTemplate().catch(showError));
function collect(){
  if(!state.brief)throw new Error("项目模板尚未加载。");
  const b=structuredClone(state.brief);for(const k of ["name","entry","route","direction"])b[k]=$(k).value;
  b.disease=$("disease").value||"unspecified";b.product.format=$("format").value;
  if(b.product.format==="VHH"){b.product.valency=1;b.product.fc="none";b.product.fc_requirement="none";b.product.chain_organization="single VHH";}
  const generator=$("generator").value;b.generators=b.route==="library"?["library"]:generator==="combined"?["rfantibody","boltzgen","iggm"]:[generator];
  b.budget.candidates=Number($("count").value);b.budget.shortlist=Math.min(4,b.budget.candidates);
  return b;
}
action("sync-json",()=>{$("brief-json").value=pretty(collect());});
async function launch(payload){const r=await api("/api/runs",payload);showView("workspace");await openRun(r.id);await loadHistory();}
action("launch",()=>launch(state.mode==="natural"?{instruction:$("instruction").value}:{brief:collect()}));
action("submit-json",()=>launch({brief:JSON.parse($("brief-json").value)}));
async function loadHistory(){const runs=await api("/api/runs");$("history").replaceChildren();if(!runs.length)$("history").textContent="暂无任务";for(const r of runs){const b=node("button",r.name);b.append(node("span",r.state+" · "+new Date(r.created*1000).toLocaleString()));b.classList.toggle("active",state.run?.id===r.id);b.addEventListener("click",()=>openRun(r.id).catch(showError));$("history").append(b);}}
action("refresh-history",loadHistory);
async function openRun(id){clearTimeout(state.timer);state.run={id};state.selected=null;viewer.load("");$("candidate-detail").textContent="选择已评估候选查看 dossier。";$("download-pdb").hidden=true;$("submit-review").disabled=true;$("review-result").textContent="尚未选择已评估候选。";$("feedback-result").textContent="读取当前项目反馈后显示。";localStorage.setItem("antibody:last-run",id);await refresh(id);}
async function refresh(id){
  const r=await api("/api/runs/"+id);if(state.run?.id!==id)return;state.run=r;render(r);
  if(!terminal.has(r.state))state.timer=setTimeout(()=>refresh(id).catch(showError),1500);
  else await loadHistory();
}
function render(r){
  if(state.selected&&!r.candidates.some(c=>c.id===state.selected)){state.selected=null;viewer.load("");$("candidate-detail").textContent="该候选已不符合当前交付条件。";$("submit-review").disabled=true;$("download-pdb").hidden=true;}
  document.querySelector('[data-stage="complete"]').textContent=r.delivery.state==="research_only"?"研究报告":"human 审阅";
  $("run-title").textContent=r.name;$("run-subtitle").textContent="项目 "+r.id+" · "+(r.research?.direction||"机制待明确");$("run-state").textContent=r.state.toUpperCase();$("run-actions").hidden=false;
  $("cancel").disabled=terminal.has(r.state);$("resume").disabled=!terminal.has(r.state)||["complete","budget_exhausted"].includes(r.state);$("reevaluate").disabled=!terminal.has(r.state)||!r.delivery.generated_count;
  for(const format of ["json","fasta","report"])$("download-"+format).href=`/api/runs/${r.id}/export/${format}`;
  for(const [name,key] of [["generated","generated_count"],["evaluated","evaluated_count"],["delivered","delivered_count"]])$("stat-"+name).textContent=r.delivery[key];$("stat-calls").textContent=r.cost.tool_calls;
  $("conclusion-title").textContent=r.delivery.state==="awaiting_human_review"?"计算评估已通过，等待 human 审阅":"交付状态 · "+r.delivery.state;
  $("conclusion").textContent=r.diagnosis||"任务执行中。全部候选会在独立评价完成后经过统一筛选。";
  document.querySelectorAll("[data-stage]").forEach(el=>el.classList.toggle("active",el.dataset.stage===r.state||(el.dataset.stage==="planning"&&r.state==="evidence_review")));
  const m=$("mechanism");m.replaceChildren();for(const [label,value] of [["疾病 / 患者",r.research.disease],["治疗假设",r.research.mechanism],["最终格式",r.research.product?.format],["靶点状态",(r.research.targets||[]).map(t=>t.entity+" · "+t.state).join("；")]]){if(value){const row=node("div",undefined,"details-row");row.append(node("b",label),node("span",value));m.append(row);}}
  if(!m.children.length)m.textContent="正在建立项目档案…";
  const gaps=$("gaps");gaps.replaceChildren();const ul=node("ul");for(const gap of r.research.gaps||[])ul.append(node("li",gap));gaps.append(ul);if(!ul.children.length)gaps.textContent="项目档案暂无新增缺项；候选性质以分项评估为准。";
  const rows=$("candidates");rows.replaceChildren();$("candidate-count").textContent=r.candidates.length+" CANDIDATES FOR REVIEW";
  for(const c of r.candidates){const tr=node("tr");tr.classList.toggle("selected",state.selected===c.id);const td=node("td"),b=node("button",c.id.slice(0,12));b.addEventListener("click",()=>choose(c.id).catch(showError));td.append(b,node("small",c.variable_id.slice(0,12)));tr.append(td,node("td",c.binder.product.format+" / "+c.binder.chains.map(x=>x.role).join("+")),node("td",c.origin));const p=r.portfolio.find(x=>x.candidate_id===c.id);tr.append(node("td",p.family+" / "+p.role));const last=node("td");last.append(node("span","PASS · 计算","tag"));tr.append(last);rows.append(tr);}
  if(!r.candidates.length){const tr=node("tr"),td=node("td","尚无可提交的合格候选。未评估和未通过候选的序列保留在本地审计记录。","empty");td.colSpan=5;tr.append(td);rows.append(tr);}
  $("evaluation-summary").textContent=pretty({delivery:r.delivery,cost:r.cost,diagnostics:r.diagnostics});$("evaluation-result").textContent=pretty(r.evaluations);
  const events=$("events");events.replaceChildren();for(const e of r.events||[]){const row=node("div",undefined,"event");row.append(node("time",new Date(e.time*1000).toLocaleTimeString()),node("span",e.kind));events.append(row);}
  if(r.reviews?.length)$("review-result").textContent=pretty(r.reviews);
}
async function choose(id){
  const run=state.run, c=run.candidates.find(x=>x.id===id);if(!c)return;state.selected=id;$("submit-review").disabled=false;render(run);
  const panel=$("candidate-detail");panel.replaceChildren();for(const [label,value] of [["不可变身份",c.id],["配对来源",c.binder.pairing_source],["格式 / Fc",c.binder.product.format+" / "+c.binder.product.fc],["父本与改动",(c.parent_id||"无父本")+" · "+c.mutations.join(", ")]]){const row=node("div",undefined,"details-row");row.append(node("b",label),node("span",value));panel.append(row);}
  for(const chain of c.binder.chains){panel.append(node("b",chain.role+" · "+chain.scheme+" · variable domain"),node("div",chain.sequence,"sequence"));}
  const evaluation=run.evaluations.find(e=>e.candidate_id===id);panel.append(node("p",evaluation.unknowns.join("；"),"hint"));
  const url=`/api/runs/${run.id}/structures/${id}`, response=await fetch(url);if(!response.ok)throw new Error("独立结构尚不可用。");const pdb=await response.text();if(state.run?.id!==run.id||state.selected!==id)return;viewer.load(pdb);$("download-pdb").href=url;$("download-pdb").hidden=false;$("download-pdb").download=id+".pdb";
}
action("cancel",async()=>{const id=requireRun();await api(`/api/runs/${id}/cancel`,{});await refresh(id);});
action("resume",async()=>{const id=requireRun();await api(`/api/runs/${id}/resume`,{});await openRun(id);});
action("reevaluate",async()=>{const r=await api(`/api/runs/${requireRun()}/reevaluate`,{});await openRun(r.id);});
action("submit-review",async()=>{if(!state.selected)throw new Error("先选择已评估候选。");const id=requireRun();const result=await api(`/api/runs/${id}/review`,{candidate_id:state.selected,decision:$("review-decision").value,note:$("review-note").value});$("review-result").textContent=pretty(result);await refresh(id);});
action("benchmark",async()=>{$("benchmark-result").textContent=pretty(await api(`/api/runs/${requireRun()}/benchmark`));});
action("feedback-template",()=>{if(!state.selected)throw new Error("请在设计工作台选择已评估候选。");const e=state.run.evaluations.find(x=>x.candidate_id===state.selected);const context=Object.values(e.contexts)[0];if(!context)throw new Error("该候选没有抗原评价上下文。");$("feedback-json").value=pretty({observations:[{id:"measurement-001",candidate_id:state.selected,context_id:context,endpoint:"binding",status:"measured",value:null,unit:"填写真实单位",censoring:"none",outcome:"undetermined",conditions:{assay:"待填写"},batch:"待填写",glycoform:"unknown",replicate:"1",control:"待填写",uncertainty:"待填写",source:"待填写原始数据来源"}]});});
action("submit-feedback",async()=>{$("feedback-result").textContent=pretty(await api(`/api/runs/${requireRun()}/feedback`,JSON.parse($("feedback-json").value)));});
action("refresh-feedback",async()=>{$("feedback-result").textContent=pretty(await api(`/api/runs/${requireRun()}/feedback`));});
async function models(){const data=await api("/api/capabilities");$("health").textContent="Qwen 0.8B · "+data.planner.status;const cards=$("model-cards");cards.replaceChildren();for(const model of data.models){const card=node("div",undefined,"model-card");card.append(node("h3",model.id),node("div",model.status,"model-status"),node("p",model.scope),node("p",model.role+" · "+(model.native_adapter?"原生适配器":"模型桥接")),node("p","代码: "+(model.version||"未配置")+" / 权重: "+(model.weights_version||"未配置")));const link=node("a","官方来源 ↗");link.href=model.source;link.target="_blank";link.rel="noopener noreferrer";card.append(link);cards.append(card);}}
action("refresh-models",models);
(async()=>{await Promise.all([loadTemplate(),loadHistory(),models()]);const last=localStorage.getItem("antibody:last-run");if(last&&/^[a-f0-9]{32}$/.test(last)){try{await openRun(last);}catch{localStorage.removeItem("antibody:last-run");}}})().catch(showError);

action("train-property",async()=>{$("training-result").textContent=pretty(await api("/api/property-models",JSON.parse($("training-json").value)));});
action("next-batch",async()=>{$("prediction-result").textContent=pretty(await api(`/api/runs/${requireRun()}/next-batch`,JSON.parse($("prediction-json").value)));});
