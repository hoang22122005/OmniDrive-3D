(function (root) {
  const G=root.Geometry, R=root.ReplayCore;
  const VIEW={yaw:-0.59,pitch:0.85,zoom:1,top:false};
  const layerColors=["#d1a275","#e7b37c","#91c3aa","#91a9bb"];
  const fill=(rgb,a=1)=>`rgba(${rgb[0]},${rgb[1]},${rgb[2]},${a})`;

  function draw(canvas, frame, view, layers, feature) {
    const w=canvas.clientWidth, h=canvas.clientHeight;
    if(w<1 || h<1) return;
    const dpr=Math.min(2,window.devicePixelRatio||1);
    const pw=Math.round(w*dpr),ph=Math.round(h*dpr);
    if(canvas.width!==pw || canvas.height!==ph){canvas.width=pw;canvas.height=ph;}
    const ctx=canvas.getContext("2d");
    ctx.setTransform(dpr,0,0,dpr,0,0);
    ctx.fillStyle="#0c1712"; ctx.fillRect(0,0,w,h);
    const scale=Math.min(w/19,h/16.6)*view.zoom;
    const yaw=view.top?0:view.yaw, pitch=view.top?Math.PI/2:view.pitch;
    const c=Math.cos(yaw), s=Math.sin(yaw), sp=Math.sin(pitch), cp=Math.cos(pitch);
    const origin=[w*0.5,h*0.51];
    function project(p) {
      const y=-p[0]*s+p[1]*c, x=p[0]*c+p[1]*s;
      return [origin[0]+scale*x,origin[1]+scale*(y*sp-p[2]*cp),y*cp+p[2]*sp];
    }
    function path(points,closed=false){
      ctx.beginPath();
      points.forEach((p,i)=>{const [x,y]=project(p); if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y);});
      if(closed)ctx.closePath();
    }
    function polygon(points,color,border){
      path(points,true);
      if(color){ctx.fillStyle=color;ctx.fill();}
      if(border){ctx.strokeStyle=border;ctx.stroke();}
    }
    function line(points,color,width=1){
      path(points);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();ctx.lineWidth=1;
    }
    function label(p,text, color="#b9d8c2"){
      const [x,y]=project(p);ctx.font="11px Segoe UI, sans-serif";
      const measure=ctx.measureText(text).width+12;
      ctx.fillStyle="#102118ee";ctx.fillRect(x+4,y-20,measure,18);
      ctx.fillStyle=color;ctx.fillText(text,x+10,y-7);
    }
    function rect(x,y,otherX,otherY,z){
      return [[x,y,z],[otherX,y,z],[otherX,otherY,z],[x,otherY,z]];
    }
    function groundTile(p,color,size=G.STEP/2) {
      polygon(rect(p[0]-size,p[1]-size,p[0]+size,p[1]+size,0.01),color);
    }
    function wire(min,max,color){
      const bottom=rect(min[0],min[1],max[0],max[1],min[2]);
      const top=rect(min[0],min[1],max[0],max[1],max[2]);
      polygon(bottom,null,color);polygon(top,null,color);
      bottom.forEach((a,i)=>line([a,top[i]],color));
    }
    function faces(obj,pose,colors,queue){
      const [a,b]=[obj.min,obj.max];
      const bottom=rect(a[0],a[1],b[0],b[1],a[2]),top=rect(a[0],a[1],b[0],b[1],b[2]);
      const list=[top,...bottom.map((p,i)=>[p,bottom[(i+1)%4],top[(i+1)%4],top[i]])];
      list.forEach((face,i)=>{
        const points=face.map(p=>G.toWorld(p,pose));
        queue.push({points,color:colors[i%colors.length],depth:points.reduce((n,p)=>n+project(p)[2],0)/points.length});
      });
    }
    function drawCar(){
      const zero={x:0,y:0,yaw:0}, queue=[];
      for(const x of [-1.28,1.3]) for(const y of [-0.89,0.89]){
        faces(G.box(x,y,0.70,0.32,0.54,"ego",0.04),zero,["#2c3b34","#0e1813","#212f28"],queue);
      }
      faces(G.box(0,0,4.2,1.88,0.60,"ego",0.31),zero,["#4b8373","#23463a","#315f51","#204337","#325e51"],queue);
      faces(G.box(-0.33,0,2.04,1.56,0.49,"ego",0.91),zero,["#70b99c","#2f5c54","#1f3d3b","#305752","#254943"],queue);
      for(const y of [-0.60,0.60]) faces(G.box(1.94,y,0.28,0.32,0.07,"ego",0.9),zero,["#d2f6cf","#bddcb9"],queue);
      queue.sort((a,b)=>a.depth-b.depth);
      queue.forEach(f=>polygon(f.points,f.color,"#83bd9a63"));
      const [x,y]=project([-.35,-.04,1.42]);
      ctx.font="700 10px Segoe UI, sans-serif";ctx.fillStyle="#112820";ctx.fillText("EGO",x-10,y+3);
    }

    polygon(rect(-8,-8,8,8,0),"#101e17","#42604d");
    for(let i=-8;i<=8;i++){
      line([[i,-8,0],[i,8,0]],i===0?"#345c48":"#253c30");
      line([[-8,i,0],[8,i,0]],i===0?"#345c48":"#253c30");
    }
    for(const i of [-6,-4,-2,2,4,6]){
      label([i,-7.8,0],`${i} m`,"#91b5a1");
    }
    label([7.25,0,0],"X · trước","#9bd9b5");
    label([0,7.5,0],"Y · trái","#9bd9b5");
    if(frame.scan.cells) {
      for(let i=0;i<G.SIZE*G.SIZE;i++){
        const states=Array.from({length:4},(_,z)=>frame.scan.cells[i+z*G.SIZE*G.SIZE]);
        if(layers.free && states[0]===G.FREE) groundTile(G.center(i),"#366c544a");
        else if(layers.shadow && (states[0]===G.SHADOW || states[1]===G.SHADOW)) groundTile(G.center(i),"#a170444c");
      }
    }
    if(layers.truth && frame.objects) for(const obj of frame.objects) {
      const corner=(x,y,z)=>G.toLocal([x,y,z],frame.pose);
      const [a,b]=[obj.min,obj.max], bottom=[
        corner(a[0],a[1],a[2]),corner(b[0],a[1],a[2]),corner(b[0],b[1],a[2]),corner(a[0],b[1],a[2])
      ];
      const top=bottom.map(p=>[p[0],p[1],b[2]]);
      polygon(bottom,null,"#96a4f080");polygon(top,null,"#96a4f080");
      bottom.forEach((p,i)=>line([p,top[i]],"#96a4f080"));
    }
    const fault=frame.features[feature];
    const amber="#ffc17b";
    if(fault && feature==="F01") {
      fault.sweep.forEach((p,i)=>{
        if(i%3 && i!==fault.sweep.length-1) return;
        const corners=rect(-2.1,-0.94,2.1,0.94,0.037).map(c=>G.toWorld(c,p));
        polygon(corners,fault.active?"#c77b3a20":"#63b78a1c",fault.active?"#e9a76175":"#82c6a66e");
      });
    }
    if(fault && feature==="F02"){
      for(const sign of [-1,1]) polygon(fault.door.map(p=>[p[0],p[1]*sign,0.04]),"#e7aa5534",amber);
    }
    if(fault && feature==="F05") polygon(rect(2.3,-1.4,4.5,2.15,0.055),"#c9894231","#cb9b66");
    if(fault && feature==="F08") {
      for(const [i,p] of fault.corridor.entries()){
        if(i%3)continue;
        polygon(rect(-0.7,-1.21,2.25,1.21,0.05).map(pt=>G.toWorld(pt,p)),
          fault.active?"#b9753928":"#64b89125",fault.active?"#ffc17b63":"#90f3be63");
      }
      if(fault.obstruction) wire([fault.obstruction[0]-.06,-1.21,0.04],[fault.obstruction[0]+.06,1.21,1.6],amber);
    }
    if(fault && feature==="F09" && fault.swing.length) line(fault.swing.map(s=>s.nose),amber,3);
    if(fault?.slot) {
      const {x,y,otherY,length,width}=fault.slot;
      polygon(rect(x,y,x+length,otherY,0.06),"#89efb440","#91f2ba");
      label([x+0.8,(y+otherY)/2,0.1],`${length.toFixed(1)} × ${width.toFixed(1)} m`);
    }
    const sorted = frame.scan.occupied.map(index=>{
      const p=G.center(index);
      return {index,p,screen:project(p)};
    }).sort((a,b)=>a.screen[2]-b.screen[2]);
    for(const {index,p,screen} of sorted){
      const [x,y]=screen, z=Math.floor(index/(G.SIZE*G.SIZE));
      const kind=G.CODES[frame.scan.hitKind?.[index]];
      const rgb=G.COLORS[kind];
      const size=G.clamp(scale*0.15,2,9);
      ctx.fillStyle=kind==="rider"||kind==="pedestrian"||kind==="stone"?fill(rgb,0.95):layerColors[z];
      ctx.globalAlpha=0.78;
      ctx.fillRect(x-size/2,y-size/2,size,size);
      ctx.globalAlpha=1;
      if(!view.top && p[2]>1.5){
        ctx.fillStyle="#d0f4d7a0";ctx.fillRect(x-size/2,y-size/2,size,1);
      }
    }
    drawCar();
    if(fault && feature==="F06"){
      polygon(rect(-2.4,-1,2.4,1,0.12),"#5ab18c30","#90f3be");
      for(const item of fault.under) groundTile(item.p,"#fec782",0.045);
      if(fault.under.length) label(fault.under[0].p,"Điểm đã lưu",amber);
    }
    if(fault?.low!==undefined) label(G.center(fault.low),"Vật thấp",amber);
    if(fault?.swingPoint) label(fault.swingPoint,"Vật trong vùng quét",amber);
    for(const a of frame.tracks || []){
      const p=a.pos, q=[p[0]+a.velocity[0]*0.6,p[1]+a.velocity[1]*0.6,p[2]];
      if(a.tracked) line([p,q],amber,2);
      const speed=Math.hypot(...a.velocity);
      label([p[0],p[1],p[2]+0.35],`${a.id} · ${speed.toFixed(1)} m/s`,a.kind==="rider"?amber:"#b6d8e9");
    }
  }
  root.DemoVisual={VIEW,draw};
})(globalThis);
