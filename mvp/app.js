(function () {
  "use strict";
  const G=window.Geometry, R=window.ReplayCore, V=window.DemoVisual;
  const $=id=>document.getElementById(id);
  const selector=$("scene"), canvas=$("world"), slider=$("timeline"), cameras=$("cameras");
  const numbers=new Intl.NumberFormat("vi-VN");
  const format=(value,digits=1)=>value.toLocaleString("vi-VN",{minimumFractionDigits:digits,maximumFractionDigits:digits});
  const faultNames={ok:"Đang dùng",mud:"Bùn che",glare:"Lóa đèn",offline:"Mất tín hiệu"};
  const replay=new R.Replay(R.SCENES[0].key);
  const view={...V.VIEW};
  const featureButtons=new Map();
  let current=0, activeFeature="F01", frame=null, timer=0, archive=null, eventFrame=null, busy=false;

  for(const scene of R.SCENES) {
    const option=document.createElement("option");
    option.value=scene.key; option.textContent=scene.name;
    selector.append(option);
  }
  for(const [id,title,description] of R.FEATURES) {
    const button=document.createElement("button");
    button.type="button";button.className="feature";button.title=description;
    for(const [value,css] of [[id,"feature-code"],[title,"feature-name"],["—","feature-state"]]){
      const span=document.createElement("span");
      span.className=css;span.textContent=value;button.append(span);
    }
    button.onclick=()=>{
      activeFeature=id;
      if(!archive && !R.SCENES.find(s=>s.key===replay.key).features.includes(id) && id!=="F11"){
        const scene=R.SCENES.find(s=>s.features.includes(id));
        if(scene) choose(scene, false);
      } else paint();
    };
    featureButtons.set(id,button);
    $("features").append(button);
  }
  const feeds=G.CAMS.map((camera,index)=>{
    const card=$("cameraTemplate").content.firstElementChild.cloneNode(true);
    card.querySelector("b").textContent=camera.name;
    const image=card.querySelector("canvas");
    image.width=G.W;image.height=G.H;
    image.setAttribute("aria-label",`Ảnh mô phỏng từ camera ${camera.name}`);
    const select=card.querySelector("select");
    select.setAttribute("aria-label",`Tình trạng camera ${camera.name}`);
    select.onchange=()=>{
      replay.fault(index,select.value,current*R.DT);
      if(eventFrame!==null && eventFrame>=current)eventFrame=null;
      const selected=select.value;
      if(selected!=="ok") activeFeature="F11";
      show(current);
      $("status").textContent=selected==="ok"?`Đã phục hồi camera ${camera.name}. Bộ lọc giữ lịch sử trong 1,5 s.`:
        `Đã đặt ${faultNames[selected].toLowerCase()} cho camera ${camera.name}.`;
    };
    cameras.append(card);
    return {card,select,context:image.getContext("2d"),status:card.querySelector(".cam-status")};
  });

  function paint(){
    if(!frame) return;
    const layers={free:$("layerFree").checked,shadow:$("layerShadow").checked,truth:$("layerTruth").checked};
    V.draw(canvas,frame,view,layers,activeFeature);
    $("view3d").setAttribute("aria-pressed",String(!view.top));
    $("view2d").setAttribute("aria-pressed",String(view.top));
    const info=frame.features[activeFeature];
    const data=R.FEATURES.find(f=>f[0]===activeFeature);
    $("overlayTitle").textContent=`${data[0]} · ${data[1]}`;
    $("overlayText").textContent=info?.summary||"Tính năng này có dữ liệu trực tiếp trong replay mô phỏng.";
    $("overlay").classList.toggle("alert",Boolean(info?.active));
    let active=0;
    featureButtons.forEach((button,id)=>{
      const status=frame.features[id]?.active;
      active+=Number(Boolean(status));
      button.classList.toggle("active",Boolean(status));
      button.setAttribute("aria-pressed",String(id===activeFeature));
      button.lastElementChild.textContent=status?(id==="F10"?"Tìm thấy":id==="F12"?"Sự kiện":"Báo"):"—";
    });
    $("alerts").textContent=`${active} trạng thái nổi bật`;
    $("voxels").textContent=numbers.format(frame.scan.counts[G.OCCUPIED]);
    $("latency").textContent=archive?"—":`${format(frame.reconstructMs)} ms`;
    $("accuracy").textContent=frame.surfaceError?`${format(frame.surfaceError.mean*100)} cm`:"—";
    $("accuracy").title=frame.surfaceError?`Cực đại ${format(frame.surfaceError.max*100)} cm · chiếu vòng ${format(frame.scan.reprojection,4)} px`:"";
    $("coverage").textContent=`${format((frame.scan.counts[1]+frame.scan.counts[2])*100/(G.SIZE*G.SIZE*4))}%`;
  }

  function show(index){
    current=G.clamp(Math.round(index),0,archive?archive.frames.length-1:R.MAX_FRAME);
    frame=archive?fromArchive(archive.frames[current]):replay.get(current);
    if(!archive && frame.features.F12.active && eventFrame===null) {
      eventFrame=current;
      $("status").textContent="Đã đánh dấu sự kiện TTC thấp. Dùng Lưu hộp đen để xuất cửa sổ 5 giây.";
    }
    slider.value=current;
    $("time").value=`${format(frame.t,2)} s`;
    if(!archive) feeds.forEach((feed,i)=>{
      const h=frame.health[i];
      feed.context.putImageData(new ImageData(frame.images[i].image,G.W,G.H),0,0);
      feed.card.classList.toggle("bad",h.gated || h.type!=="ok");
      feed.select.value=h.type;
      feed.status.textContent=h.gated? "Đã tách" : h.type==="ok" ? "Đang dùng" : "Đang lọc";
    });
    paint();
  }

  function pause(){
    window.clearInterval(timer);timer=0;$("play").textContent="Phát";
  }

  function choose(scene,selectFeature=true){
    pause();archive=null;eventFrame=null;
    replay.reset(scene.key);selector.value=scene.key;
    if(selectFeature) activeFeature=scene.features[0];
    slider.max=R.MAX_FRAME;cameras.hidden=false;
    $("viewSource").textContent="Voxel từ 4 fisheye RGB-D";
    $("record").disabled=false;$("status").textContent="";
    show(scene.start/R.DT);
  }

  function play(){
    if(timer){pause();return;}
    const last=Number(slider.max);
    if(current>=last){eventFrame=null;show(0);}
    $("play").textContent="Tạm dừng";
    timer=window.setInterval(()=>{
      if(current>=last){pause();return;}
      show(current+1);
    },1000*R.DT);
  }

  function fromArchive(entry){
    const cells=new Uint8Array(G.SIZE*G.SIZE*4), counts=[0,0,0,0],occupied=[];
    let index=0;
    for(let i=0;i<entry.state.length;i+=2) {
      const value=entry.state[i],length=entry.state[i+1];
      cells.fill(value,index,index+length);
      counts[value]+=length;
      if(value===G.OCCUPIED) for(let j=index;j<index+length;j++)occupied.push(j);
      index+=length;
    }
    const ttc=entry.ttc;
    return {t:entry.t,scan:{cells,counts,occupied},tracks:entry.tracks,
      features:{F12:{active:ttc!==null&&ttc<1.2,summary:ttc===null?"Không có TTC được ghi":`TTC ghi nhận ${format(ttc,2)} s`}}};
  }

  function validArchive(log){
    if (log?.format!=="omnishield-replay/1" || log.grid?.size!==G.SIZE || log.grid.resolution!==G.STEP ||
      !Array.isArray(log.grid.z) || log.grid.z.length!==G.Z.length ||
      !G.Z.every((value,index)=>log.grid.z[index]===value) ||
      !Array.isArray(log.frames) || log.frames.length!==41) return false;
    const vector=(value,size)=>Array.isArray(value)&&value.length===size&&value.every(Number.isFinite);
    return log.frames.every((f,index)=>{
      if (!Number.isFinite(f?.t) || (index && f.t<=log.frames[index-1].t) ||
        !(f.ttc===null || Number.isFinite(f.ttc)) || !Array.isArray(f.state) || f.state.length%2 ||
        !Array.isArray(f.tracks) || f.tracks.length>10) return false;
      let total=0;
      for(let k=0;k<f.state.length;k+=2) {
        const v=f.state[k], n=f.state[k+1];
        if(!Number.isInteger(v)||v<0||v>3||!Number.isInteger(n)||n<1)return false;
        total+=n;
        if(total>G.SIZE*G.SIZE*4)return false;
      }
      return total===G.SIZE*G.SIZE*4 && f.tracks.every(a=>typeof a.id==="string"&&a.id.length<=12&&
        vector(a.pos,3)&&vector(a.velocity,2));
    });
  }

  $("record").onclick=()=>{
    if(archive || busy)return;
    busy=true;pause();$("record").disabled=true;$("status").textContent="Đang mã hóa 41 khung hình…";
    window.setTimeout(()=>{
      try {
        const data=replay.clip(eventFrame??current);
        const blob=new Blob([JSON.stringify(data)],{type:"application/json"});
        const url=URL.createObjectURL(blob), anchor=document.createElement("a");
        anchor.href=url; anchor.download=`omnishield-${data.scene}-${current}.json`; anchor.click();
        window.setTimeout(()=>URL.revokeObjectURL(url),1000);
        $("status").textContent=`Đã xuất ${format(blob.size/1024)} KiB · 5 s · điểm demo ${data.score}/100.`;
      } finally { busy=false;$("record").disabled=false; }
    },30);
  };

  $("openLog").onclick=()=>$("logFile").click();
  $("logFile").onchange=async e=>{
    const file=e.target.files[0];e.target.value="";
    if(!file)return;
    try {
      if(file.size>15_000_000) throw new Error("size");
      const data=JSON.parse(await file.text());
      if(!validArchive(data))throw new Error("format");
      pause();archive=data;activeFeature="F12";slider.max=data.frames.length-1;
      cameras.hidden=true;$("record").disabled=true;$("viewSource").textContent="Dữ liệu từ hộp đen 3D";
      selector.value="";
      $("status").textContent="Đã mở 5 giây trạng thái voxel. Chọn kịch bản để trở lại camera mô phỏng.";
      show(0);
    } catch {
      $("status").textContent="Tệp chưa đúng định dạng hộp đen OmniShield hoặc vượt quá 15 MB.";
    }
  };
  selector.onchange=()=>choose(R.SCENES.find(s=>s.key===selector.value));
  slider.oninput=()=>{pause();show(Number(slider.value));};
  $("play").onclick=play;
  $("view3d").onclick=()=>{view.top=false;paint();};
  $("view2d").onclick=()=>{view.top=true;paint();};
  $("viewReset").onclick=()=>{Object.assign(view,V.VIEW);paint();};
  for(const id of ["layerFree","layerShadow","layerTruth"])$(id).onchange=paint;

  let drag=null;
  canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);};
  canvas.onpointermove=e=>{
    if(!drag)return;
    view.top=false;view.yaw+=(e.clientX-drag[0])*0.008;
    view.pitch=G.clamp(view.pitch+(e.clientY-drag[1])*0.006,0.32,1.54);
    drag=[e.clientX,e.clientY];paint();
  };
  canvas.onlostpointercapture=()=>{drag=null;};
  canvas.addEventListener("wheel",e=>{
    e.preventDefault();view.zoom=G.clamp(view.zoom*Math.exp(-e.deltaY*0.001),0.65,2.8);paint();
  },{passive:false});
  canvas.onkeydown=e=>{
    const keys={ArrowLeft:[-0.15,0],ArrowRight:[0.15,0],ArrowUp:[0,0.12],ArrowDown:[0,-0.12]};
    if(!keys[e.key])return;
    e.preventDefault();view.top=false;
    view.yaw+=keys[e.key][0];view.pitch=G.clamp(view.pitch+keys[e.key][1],0.32,1.54);
    paint();
  };
  document.addEventListener("visibilitychange",()=>{if(document.hidden)pause();});
  new ResizeObserver(paint).observe(canvas);
  choose(R.SCENES[0]);
})();
