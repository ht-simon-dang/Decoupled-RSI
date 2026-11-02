'use strict';
// Local canvas projection, following the Enzyme Agent viewer. No remote scripts.
class StructureViewer {
  constructor(container, runId, candidateId, initialKind='rna', onKind=()=>{}) {
    this.container=container;this.path='/api/runs/'+encodeURIComponent(runId)+'/preview/candidates/'+encodeURIComponent(candidateId);
    this.onKind=onKind;this.kind=initialKind;this.token=0;this.closed=false;this.points=[];this.edges=[];this.timer=null;
    const make=(tag,text,cls)=>{const e=document.createElement(tag);if(text)e.textContent=text;if(cls)e.className=cls;return e;};
    const button=(text,cls,action)=>{const b=make('button',text,cls);b.type='button';b.addEventListener('click',action);return b;};
    container.className='structure-viewer';
    const heading=make('div',null,'panel-heading');heading.append(make('h3','结构预览'));
    const tabs=make('div',null,'structure-tabs');
    this.rnaButton=button('RNA 二级结构示意','rna-tab',()=>this.load('rna'));
    this.proteinButton=button('编码蛋白 3D','protein-tab',()=>this.load('protein'));
    tabs.append(this.rnaButton,this.proteinButton);heading.append(tabs);container.append(heading);
    this.message=make('p','加载结构证据…','structure-message');this.message.setAttribute('role','status');container.append(this.message);
    const toolbar=make('div',null,'actions structure-toolbar');
    toolbar.append(button('复位','reset-view',()=>{this.reset();this.draw();}),
      button('放大 +','zoom-in',()=>this.zoomBy(1.2)),button('缩小 −','zoom-out',()=>this.zoomBy(1/1.2)),
      button('全屏','fullscreen-view',async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await container.requestFullscreen();}catch{this.message.textContent='当前浏览器不支持全屏；仍可拖动和缩放。';}}));
    this.generate=button('生成蛋白预测结构','generate-structure primary',()=>this.predict());this.generate.hidden=true;
    this.download=make('a','下载蛋白 PDB','protein-download');this.download.href=this.path+'/protein.pdb';this.download.hidden=true;
    toolbar.append(this.generate,this.download);container.append(toolbar);
    this.canvas=make('canvas',null,'structure-canvas');this.canvas.tabIndex=0;this.canvas.setAttribute('role','img');
    this.canvas.setAttribute('aria-label','结构交互预览；拖动或方向键旋转，滚轮或加减键缩放，0 键复位');container.append(this.canvas);
    this.legend=make('div',null,'structure-legend');container.append(this.legend);
    this.info=make('p','拖动旋转 · 滚轮缩放 · 点击查看位置','structure-position');container.append(this.info);
    this.caption=make('p',null,'hint structure-caption');container.append(this.caption);
    this.abort=new AbortController();const options={signal:this.abort.signal};let pointer=null;
    window.addEventListener('rsi-theme-change',()=>this.draw(),options);
    this.canvas.addEventListener('pointerdown',e=>{pointer={id:e.pointerId,x:e.clientX,y:e.clientY};this.canvas.setPointerCapture(e.pointerId);},options);
    this.canvas.addEventListener('pointermove',e=>{if(pointer?.id===e.pointerId){this.angle+=(e.clientX-pointer.x)*.012;this.tilt+=(e.clientY-pointer.y)*.012;pointer.x=e.clientX;pointer.y=e.clientY;this.draw();}this.inspect(e);},options);
    this.canvas.addEventListener('pointerup',()=>{pointer=null;},options);
    this.canvas.addEventListener('pointercancel',()=>{pointer=null;},options);
    this.canvas.addEventListener('wheel',e=>{e.preventDefault();this.zoomBy(Math.exp(-e.deltaY*.001));},{...options,passive:false});
    this.canvas.addEventListener('keydown',e=>{const changes={ArrowLeft:[-.12,0],ArrowRight:[.12,0],ArrowUp:[0,-.12],ArrowDown:[0,.12]};
      if(changes[e.key]){e.preventDefault();this.angle+=changes[e.key][0];this.tilt+=changes[e.key][1];this.draw();}
      if(['+','=','-','0'].includes(e.key)){e.preventDefault();if(e.key==='0'){this.reset();this.draw();}else this.zoomBy(e.key==='-'?1/1.2:1.2);}},options);
    this.observer=new ResizeObserver(()=>this.draw());this.observer.observe(this.canvas);this.reset();this.load(initialKind);
  }
  reset(){this.angle=this.kind==='rna'?0:.5;this.tilt=this.kind==='rna'?0:.3;this.zoom=1;}
  zoomBy(factor){this.zoom=Math.max(.2,Math.min(8,this.zoom*factor));this.draw();}
  async request(suffix,method='GET'){
    const r=await fetch(this.path+'/'+suffix,{method,signal:this.abort.signal});const data=await r.json();
    if(!r.ok)throw Error(data.detail||'无法读取结构');return data;
  }
  current(token){return !this.closed&&token===this.token;}
  async load(kind){
    this.kind=kind;this.onKind(kind);this.canvas.dataset.status='loading';this.canvas.dataset.kind=kind;delete this.canvas.dataset.pointCount;const token=++this.token;clearTimeout(this.timer);this.points=[];this.edges=[];this.data=null;this.reset();this.draw();
    this.rnaButton.classList.toggle('active',kind==='rna');this.proteinButton.classList.toggle('active',kind==='protein');
    this.generate.hidden=true;this.download.hidden=true;this.legend.replaceChildren();this.caption.textContent='';this.message.textContent='加载结构证据…';
    try{const data=await this.request(kind);if(this.current(token))this.display(data,token);}catch(e){this.error(e,token);}
  }
  error(error,token){if(this.current(token)&&error.name!=='AbortError'){this.message.textContent='结构预览失败：'+error.message;this.canvas.dataset.status='failed';}}
  async predict(){
    const token=this.token;this.generate.disabled=true;
    try{const data=await this.request('protein','POST');if(this.current(token))this.display(data,token);}catch(e){this.error(e,token);}finally{this.generate.disabled=false;}
  }
  display(data,token){
    if(!this.current(token))return;
    this.canvas.dataset.status=data.status;this.canvas.dataset.kind=this.kind;this.data=data;
    this.generate.hidden=this.kind!=='protein'||!['not_generated','failed'].includes(data.status);
    this.generate.textContent=data.status==='failed'?'重试蛋白结构预测':'生成蛋白预测结构';this.download.hidden=data.status!=='ready'||this.kind!=='protein';
    if(data.status==='running'){
      this.message.textContent='蛋白结构正在排队或预测，可切换任务，完成后会保留结果。';
      this.timer=setTimeout(async()=>{try{const result=await this.request('protein');if(this.current(token))this.display(result,token);}catch(e){this.error(e,token);}},2000);return;
    }
    if(data.status!=='ready'){
      this.message.textContent=data.status==='not_generated'?'尚无编码蛋白的预测坐标。点击生成后可在线旋转、缩放和检查置信度。':
        (data.status==='failed'?'结构预测失败，可重试。':'该结构暂不可用。');
      this.caption.textContent=data.reason||'仅预测 CDS 编码的蛋白；相同蛋白共享结果。RNA 拓扑仍可直接查看。';this.draw();return;
    }
    const mean=[0,1,2].map(axis=>data.points.reduce((sum,p)=>sum+p[axis],0)/data.points.length);
    this.points=data.points.map(p=>p.map((v,i)=>v-mean[i]));
    this.extent=Math.max(1,...this.points.map(p=>Math.hypot(...p)));
    this.edges=this.points.slice(1).map((_,i)=>[i,i+1]);
    if(this.kind==='rna'){
      this.message.textContent=`${this.points.length} nt · ${data.pairs.length} 对碱基配对 · 历史评价中的 MFE 结构`;
      this.caption.textContent='RNA 二级结构的平面拓扑，可在三维视角中旋转；坐标为布局单位，并非 RNA 原子级三维预测。';
      this.legend.textContent='● 5′UTR（琥珀）  ● CDS（青绿）  ● 3′UTR（紫）  ● poly(A)（灰） · 细线为碱基配对';
    }else{
      this.message.textContent=`${this.points.length} aa · ESMFold2-Fast · 平均 pLDDT ${(data.mean_plddt_fraction*100).toFixed(1)} / 100 · pTM ${data.ptm.toFixed(3)}`;
      this.caption.textContent='CDS 编码蛋白的真实预测坐标，显示单链 Cα 骨架；同义序列共享相同蛋白结构。未预测 RNA、复合物或修饰；不作为 mRNA 高潜力筛选证据。';
      this.legend.textContent='pLDDT：≥90 蓝 · 70–90 青 · 50–70 黄 · <50 橙（低置信度）';
    }
    this.canvas.dataset.pointCount=String(this.points.length);this.draw();
  }
  color(i){
    if(this.kind==='protein'){const p=this.data.plddt_fraction[i];return p>=.9?'#3475c5':p>=.7?'#24a8ae':p>=.5?'#d7b442':'#db7b35';}
    const b=this.data.boundaries;return i<b.utr5[1]?'#b98a38':i<b.cds[1]?'#168b80':i<b.utr3[1]?'#9272bf':'#84949b';
  }
  draw(){
    if(this.closed)return;
    const rect=this.canvas.getBoundingClientRect();if(!rect.width||!rect.height)return;
    const dpr=Math.min(devicePixelRatio||1,2),w=rect.width,h=rect.height;
    this.canvas.width=Math.round(w*dpr);this.canvas.height=Math.round(h*dpr);
    const ctx=this.canvas.getContext('2d');ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);
    const caption=getComputedStyle(this.canvas).getPropertyValue('--suite-muted').trim() || '#718383';
    if(!this.points.length){ctx.fillStyle=caption;ctx.font='13px system-ui';ctx.textAlign='center';ctx.fillText('等待结构坐标',w/2,h/2);return;}
    const scale=Math.min(w,h)*.44/this.extent*this.zoom,ca=Math.cos(this.angle),sa=Math.sin(this.angle),ct=Math.cos(this.tilt),st=Math.sin(this.tilt);
    this.projected=this.points.map(([x,y,z])=>{const rx=x*ca+z*sa,rz=-x*sa+z*ca;return [w/2+rx*scale,h/2+(y*ct-rz*st)*scale,y*st+rz*ct];});
    const line=(a,b,color,width)=>{const p=this.projected[a],q=this.projected[b];ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.stroke();};
    if(this.kind==='rna')for(const [a,b]of this.data.pairs)line(a,b,'#bccfc8',.8);
    ctx.lineCap='round';
    for(const [a,b]of [...this.edges].sort((a,b)=>this.projected[a[0]][2]-this.projected[b[0]][2]))line(a,b,this.color(b),this.kind==='protein'?3:1.3);
    for(const i of this.points.map((_,i)=>i).sort((a,b)=>this.projected[a][2]-this.projected[b][2])){
      const p=this.projected[i];ctx.fillStyle=this.color(i);ctx.beginPath();ctx.arc(p[0],p[1],this.kind==='protein'?2.3:1.5,0,Math.PI*2);ctx.fill();
    }
    ctx.fillStyle=caption;ctx.font='11px system-ui';
    for(const [i,label]of [[0,this.kind==='rna'?"5′":'N'],[this.points.length-1,this.kind==='rna'?"3′":'C']]){const p=this.projected[i];ctx.fillText(label,p[0]+6,p[1]-6);}
    this.canvas.dataset.camera=[this.angle,this.tilt,this.zoom].map(v=>v.toFixed(3)).join(',');
  }
  inspect(event){
    if(!this.points.length||!this.projected)return;
    const rect=this.canvas.getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top;
    let index=-1,distance=14;
    this.projected.forEach((p,i)=>{const d=Math.hypot(p[0]-x,p[1]-y);if(d<distance){distance=d;index=i;}});
    if(index<0)return;
    this.info.textContent=this.kind==='protein'?`残基 ${index+1} · ${this.data.sequence[index]} · pLDDT ${(this.data.plddt_fraction[index]*100).toFixed(1)} / 100`:
      `核苷酸 ${index+1} · ${this.data.sequence[index]} · ${this.data.dot_bracket[index]==='.'?'未配对':'配对'}（位置从 1 开始）`;
  }
  dispose(){this.closed=true;clearTimeout(this.timer);this.abort.abort();this.observer.disconnect();}
}
window.StructureViewer=StructureViewer;
