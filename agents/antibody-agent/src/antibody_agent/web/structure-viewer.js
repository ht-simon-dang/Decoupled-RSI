"use strict";
// Small local CA viewer. No external scripts or molecular data uploads.
window.AntibodyStructure = class {
  constructor(canvas) {
    this.canvas = canvas; this.points = []; this.angle = 0; this.tilt = -.25; this.zoom = 1;
    let previous = null;
    canvas.addEventListener("pointerdown", event => {previous = [event.clientX, event.clientY]; canvas.setPointerCapture(event.pointerId);});
    canvas.addEventListener("pointermove", event => {if (!previous) return; this.angle += (event.clientX-previous[0])*.012; this.tilt += (event.clientY-previous[1])*.012; previous = [event.clientX,event.clientY]; this.draw();});
    canvas.addEventListener("pointerup", () => {previous = null;});
    canvas.addEventListener("pointercancel", () => {previous = null;});
    canvas.addEventListener("wheel", event => {event.preventDefault(); this.zoom = Math.max(.4,Math.min(3,this.zoom*Math.exp(-event.deltaY*.001))); this.draw();}, {passive:false});
    new ResizeObserver(() => this.draw()).observe(canvas);
    window.addEventListener('rsi-theme-change', () => this.draw());
    this.draw();
  }
  load(pdb) {
    this.points = pdb.split("\n").filter(row => row.startsWith("ATOM") && row.slice(12,16).trim() === "CA").map(row => ({chain:row[21], xyz:[Number(row.slice(30,38)),Number(row.slice(38,46)),Number(row.slice(46,54))]})).filter(p => p.xyz.every(Number.isFinite));
    if (this.points.length) {
      const center = [0,1,2].map(i => this.points.reduce((sum,p) => sum+p.xyz[i],0)/this.points.length);
      this.points.forEach(p => {p.xyz = p.xyz.map((x,i) => x-center[i]);});
    }
    this.zoom = 1; this.draw();
  }
  draw() {
    const c = this.canvas, width = c.clientWidth || 680, height = c.clientHeight || 280, ratio = window.devicePixelRatio || 1;
    c.width = width*ratio; c.height = height*ratio;
    const ctx = c.getContext("2d"); ctx.scale(ratio,ratio); ctx.clearRect(0,0,width,height);
    if (!this.points.length) {ctx.fillStyle=getComputedStyle(c).getPropertyValue('--suite-muted').trim() || '#657a87'; ctx.font="12px system-ui"; ctx.textAlign="center"; ctx.fillText("独立预测结构将在评估通过后显示",width/2,height/2); return;}
    const radius = Math.max(1,...this.points.map(p => Math.hypot(...p.xyz)));
    const scale = Math.min(width,height)*.4/radius*this.zoom;
    const projected = this.points.map(p => {
      const [x,y,z] = p.xyz, a = this.angle, t = this.tilt, xx = x*Math.cos(a)+z*Math.sin(a), zz = -x*Math.sin(a)+z*Math.cos(a);
      return {chain:p.chain, x:width/2+xx*scale, y:height/2+(y*Math.cos(t)-zz*Math.sin(t))*scale};
    });
    ctx.lineWidth=2.4; ctx.lineJoin="round"; ctx.lineCap="round";
    for (let i=1;i<projected.length;i++) {const a=projected[i-1],b=projected[i]; if(a.chain!==b.chain)continue; ctx.strokeStyle=({H:"#2679a9",L:"#27a298",A:"#af9b79"})[b.chain]||"#657a87"; ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();}
  }
};
