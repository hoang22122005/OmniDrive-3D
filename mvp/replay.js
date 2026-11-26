(function (root) {
  const G = typeof module !== "undefined" ? require("./geometry") : root.Geometry;
  const {box,clamp,center,indices,futurePose,toLocal,toWorld,SIZE,MIN,STEP,Z,OCCUPIED,FREE,SHADOW} = G;
  const DT = 0.125, DURATION = 10, MAX_FRAME = Math.round(DURATION/DT);
  const EGO = {halfLength:2.1, halfWidth:0.94, height:1.5};
  const FEATURES = [
    ["F01","Lá chắn rẽ chậm","Dự báo xe máy hoặc người cắt đa giác quét khi xe rẽ."],
    ["F02","Bảo vệ mở cửa","Đo giao cắt giữa quỹ đạo vật thể và vùng cửa khi xe chạy chậm."],
    ["F03","Xe máy tạt đầu","Theo dõi vận tốc ngang của xe máy và dự báo điểm nhập làn."],
    ["F04","Phanh nhìn trước và sau","So các ứng viên gia tốc trong mô phỏng, có khoảng trống trước và sau."],
    ["F05","Cảnh báo góc khuất","Đo bóng che trong hành lang và hiển thị điểm rủi ro hình học."],
    ["F06","Trí nhớ 3D dưới gầm","Giữ điểm thấp đã quan sát trong hệ thế giới và chiếu về dưới xe."],
    ["F07","Vật thấp lúc xuất phát","Quét vật thấp quanh thân khi xe mô phỏng bắt đầu chuyển động."],
    ["F08","Thảm dẫn đường hẻm","Đối chiếu quỹ đạo và khoảng cao của vật cản, vẽ hành lang 3D."],
    ["F09","Chống văng đầu khi lùi","Kiểm tra phần đầu xe quét ra ngoài trong khi lùi và đánh lái."],
    ["F10","Tìm ô đỗ không cần vạch","Tìm khoảng trống đã quan sát giữa hai cụm vật cản bên sườn."],
    ["F11","Sức khỏe ống kính","Giảm độ phủ khi camera bẩn, chói hoặc mất tín hiệu."],
    ["F12","Hộp đen 3D","Lưu và mở lại cửa sổ trạng thái quanh sự kiện TTC thấp."]
  ];
  const SCENES = [
    {key:"mixed", name:"Xe máy tạt đầu", features:["F01","F03","F04","F12"], start:2},
    {key:"door", name:"Xe máy sát cửa", features:["F02"], start:2.5},
    {key:"blind", name:"Góc khuất sau xe đỗ", features:["F05","F11"], start:2},
    {key:"under", name:"Vật cản dưới gầm", features:["F06"], start:4},
    {key:"start", name:"Vật thấp lúc xuất phát", features:["F07"], start:2.5},
    {key:"alley", name:"Ngõ hẹp và mái thấp", features:["F08"], start:1.5},
    {key:"reverse", name:"Lùi và văng đầu", features:["F09"], start:1},
    {key:"slot", name:"Ô đỗ không vạch", features:["F10"], start:0}
  ];

  const hits = (p,bounds) => p[0]>=bounds[0] && p[0]<=bounds[1] && p[1]>=bounds[2] && p[1]<=bounds[3];
  const isVRU = track => track.kind === "rider" || track.kind === "pedestrian";
  const ahead = (p, t, velocity) => [p[0]+t*velocity[0], p[1]+t*velocity[1], p[2]];
  const pathTimes = Array.from({length:16},(_,i)=>(3+i)/10);
  const bodyBounds = [-EGO.halfLength,EGO.halfLength,-EGO.halfWidth,EGO.halfWidth];
  const distanceToBody = ([x,y]) => Math.hypot(Math.max(0,Math.abs(x)-EGO.halfLength), Math.max(0,Math.abs(y)-EGO.halfWidth));

  function rider(x,y) {
    return [box(x,y,1.48,0.52,0.57,"rider",0.08),box(x-0.16,y,0.52,0.43,0.92,"rider",0.65)];
  }

  function pedestrian(x,y,height) {
    return [box(x,y,0.29,0.36,height*0.73,"pedestrian",0.05),box(x+0.01,y,0.23,0.24,height*0.29,"pedestrian",height*0.72)];
  }

  function parked(x,y,kind="car") {
    const h=kind==="truck"?2.2:1.05;
    return [box(x,y,3.7,1.58,h,kind,0.06),box(x-0.35,y,1.8,1.46,0.44,kind,h)];
  }

  function sceneAt(key,t) {
    let pose = {x:0,y:0,yaw:0}, v=0, yawRate=0;
    const objects = [
      box(10.2,5.8,10,2,3.7,"wall"),box(-2.5,-6.9,24,3,3.5,"wall"),
      box(0,-4.6,27,0.25,0.13,"curb"),box(0,4.55,27,0.24,0.13,"curb"),
      box(9,-4.3,0.4,0.4,1.8,"plant"),box(-7,4,0.65,0.65,1.4,"plant")
    ];
    if(key==="mixed") {
      v=1.45; yawRate=-0.32; pose.x=t*v;
      objects.push(...rider(0.57+2.18*t,clamp(-3.5+(t-0.35)*1.02,-4,2)),
        ...parked(-6.1+1.85*t,0.12,"follower"),...parked(2.5,3.2),box(6.8,-3.3,0.8,0.9,0.82,"plant"));
    } else if(key==="door") {
      objects.push(...rider(-7+2.2*t,1.56),...parked(3.7,-3.3),box(0,4.45,10,0.2,0.22,"curb"));
    } else if(key==="blind") {
      v=0.95; pose.x=t*v*0.35;
      objects.push(...parked(4.25,1.41,"truck"),...rider(5.2,4.8-t*0.56),box(4.85,-2.45,2,1.1,0.7,"wall"));
    } else if(key==="under") {
      v=0.8; pose.x=t*v;
      objects.push(box(3.8,0.72,0.37,0.34,0.29,"stone"),...parked(4,-3.25));
    } else if(key==="start") {
      v=t<1?0:1.13; pose.x=0.18*Math.max(0,t-1);
      objects.push(...pedestrian(3.4,-0.67,0.78),box(3.5,1.1,0.34,0.32,0.37,"stone"),...parked(-1,-3.2));
    } else if(key==="alley") {
      v=1.35; yawRate=0.13; pose.x=0.31*t;
      objects.push(box(3.9,1.74,11,0.46,2.9,"wall"),box(4.5,-1.68,10,0.45,2.8,"wall"),
        box(4.55,0,1.1,3,0.35,"wall",1.43),box(3.65,-1.16,0.6,0.6,0.85,"plant"));
    } else if(key==="reverse") {
      v=-1.35; yawRate=0.65; pose.x=-0.15*t;
      objects.push(box(1.05,1.53,0.54,0.46,1.02,"plant"),...parked(-4,-2.4),box(-5.5,2,0.45,0.46,1.1,"wall"));
    } else {
      v=0.5; pose.x=0.12*t;
      objects.push(...parked(-4.8,-2.9),...parked(4.8,-2.9),box(1,3.3,1,0.86,0.9,"plant"));
    }
    return {pose,v,yawRate,objects};
  }

  function healthAt(faults, t) {
    return faults.map(events=>{
      let type="ok", integral=0;
      for(let i=0;i<events.length && events[i].at<=t;i++) {
        const f=events[i];
        type=f.type;
        if(type!=="ok") integral+=0.62*Math.max(0,Math.min(t,events[i+1]?.at??t)-Math.max(t-1.5,f.at));
      }
      const blocked=type==="ok"?0:0.62, mean=integral/1.5, offline=type==="offline";
      return {type,blocked,mean,offline,gated:offline || mean>0.25};
    });
  }

  function trackDetections(detections, previous, pose) {
    return detections.map(d=>{
      const world=toWorld(d.pos,pose);
      const match=previous?.tracks.find(p=>p.kind===d.kind);
      const last = match ? toLocal(match.world,pose) : d.pos;
      const raw=last.slice(0,2).map((p,i)=>(d.pos[i]-p)/DT);
      const velocity=raw.map((v,i)=>{
        const expected=match?match.velocity[i]:0;
        return clamp(v,-8,8)*0.55+expected*0.45;
      });
      return {...d,world,velocity,id:d.kind==="rider"?"R01":d.kind==="car"?"V01":"P01",tracked:Boolean(match)};
    });
  }

  function memoryFor(scan,previous,pose,t) {
    const memory = new Map();
    for (const item of previous?.memory || []) {
      const p = toLocal(item.world,pose);
      if (t-item.seen<=3.5 && p[0]>-2.45 && Math.abs(p[1])<=1.05) memory.set(item.key,{...item,p});
    }
    for (const p of scan.lowPoints) {
      if(p[0]<-2.4 || p[0]>4.8 || Math.abs(p[1])>1.05) continue;
      const world=toWorld(p,pose), key=world.slice(0,2).map(v=>Math.floor(v/0.05)).join(":");
      const old=memory.get(key);
      if(!old || p[2]>=old.p[2]) memory.set(key,{key,world,p,seen:t});
    }
    return [...memory.values()];
  }

  function riskFor(frame) {
    const {scan,tracks,v,yawRate,memory,health,context} = frame;
    const result={};
    const put=(id,active,summary,detail={})=>{
      result[id]={active:Boolean(active),summary,...detail};
    };
    const sweep=pathTimes.map(t=>({t,...futurePose(v,yawRate,t)}));
    const hitsByTime=sweep.map(s=>tracks.find(a=>{
      if(!isVRU(a) || !a.tracked)return false;
      const p=toLocal(ahead(a.pos,s.t,a.velocity),s);
      const body=a.kind==="rider"?[0.74,0.26]:[0.16,0.18];
      const yaw=Math.atan2(a.velocity[1],a.velocity[0])-s.yaw;
      const halfX=Math.abs(Math.cos(yaw))*body[0]+Math.abs(Math.sin(yaw))*body[1];
      const halfY=Math.abs(Math.sin(yaw))*body[0]+Math.abs(Math.cos(yaw))*body[1];
      return Math.abs(p[0])<=EGO.halfLength+halfX+0.1 && Math.abs(p[1])<=EGO.halfWidth+halfY+0.1;
    }));
    const first=hitsByTime.findIndex(Boolean);
    const ttc=first<0?null:sweep[first].t;
    put("F01",v>0 && Math.abs(yawRate)>0.13 && ttc!==null,ttc===null?"Chưa có giao cắt dự báo":`Giao cắt sau ${ttc.toFixed(2)} s`,{sweep});

    const door = tracks.find(a=>isVRU(a) && a.tracked && pathTimes.some(t=>{
      const p=ahead(a.pos,t,a.velocity);
      return p[0]>=-1.35 && p[0]<=1.45 && Math.abs(p[1])>0.94 && Math.abs(p[1])<2.15;
    }));
    put("F02",context==="door" && Math.abs(v)<0.3 && door,`Vùng mở cửa ${door?"có phương tiện đi vào":"chưa có đường cắt"}`,
      {door:[[-1.35,0.94],[1.45,0.94],[1.45,2.15],[-1.35,2.15]]});

    const cut=tracks.find(a=>a.kind==="rider" && a.tracked && a.pos[0]>0.1 && a.pos[0]<6 &&
      Math.abs(a.pos[1])>0.9 && a.velocity[1]*Math.sign(a.pos[1])<-0.35 &&
      pathTimes.some(t=>Math.abs(a.pos[1]+t*a.velocity[1])<1.12));
    put("F03",cut,cut?`${cut.id} tiến vào trục xe`:"Chưa thấy động lượng hướng vào làn");

    const front=tracks.filter(a=>isVRU(a) && a.pos[0]>1.65 && (Math.abs(a.pos[1])<1.75 || a===cut || hitsByTime[first]===a))
      .sort((a,b)=>a.pos[0]-b.pos[0])[0];
    const rear=tracks.find(a=>a.kind==="car" && a.pos[0]<-2.1 && Math.abs(a.pos[1])<1.4);
    let braking=null;
    if((ttc!==null || cut) && front && rear?.tracked && v>0) {
      const candidates=[0,-0.8,-1.6,-2.4,-3.2,-4.4].map(a=>{
        const stop=a<0?v/-a:Infinity;
        let frontGap=Infinity,rearGap=Infinity,score=0.025*a*a;
        for(const t of pathTimes){
          const tMove=Math.min(t,stop), distance=v*tMove+a*tMove*tMove/2;
          const pose=futurePose(1,yawRate/v,distance);
          const projected=toLocal(ahead(front.pos,t,front.velocity),pose);
          const following=toLocal(ahead(rear.pos,t,rear.velocity),pose);
          if(Math.abs(projected[1])<1.6 && projected[0]>-0.3){
            const gap=projected[0]-2.7;
            frontGap=Math.min(frontGap,gap);
            score+=4*Math.max(0,0.55-gap)**2;
          }
          if(Math.abs(following[1])<1.6 && following[0]<0){
            const gap=-following[0]-2.9;
            rearGap=Math.min(rearGap,gap);
            score+=1.2*Math.max(0,0.55-gap)**2;
          }
        }
        return {a,frontGap,rearGap,score};
      });
      braking=candidates.sort((a,b)=>a.score-b.score)[0];
    }
    const overlap=braking && Math.min(braking.frontGap,braking.rearGap)<0;
    put("F04",braking,braking?`Ứng viên ${braking.a.toFixed(1)} m/s² · ${overlap?"vẫn có giao cắt dự báo":"đã kiểm khoảng trống hai phía"}`:
      rear?"Chờ giao cắt phía trước":"Thiếu quan sát xe phía sau",{braking});

    let shadow=0, known=0;
    for(let x=2.3;x<4.5;x+=STEP) for(let y=-1.4;y<=2.15;y+=STEP) {
      const i=indices([x,y,1.3]);
      if(scan.cells[i]===SHADOW) shadow++;
      if(scan.cells[i]!==0) known++;
    }
    const blind=known?shadow/known:0;
    put("F05",v>0 && shadow>=5 && blind>0.35,`Bóng che ${shadow} ô · tỷ lệ ${Math.round(blind*100)}%`,{blind});

    const under=memory.filter(a=>Math.abs(a.p[0])<2.4 && Math.abs(a.p[1])<=1.0);
    const underRisk=under.some(a=>Math.abs(a.p[1])>0.62 || a.p[2]>0.20);
    put("F06",underRisk,under.length?`${under.length} điểm thấp trong vùng gầm`:"Chưa có điểm thấp lưu dưới gầm",{under});
    const low=scan.occupied.find(i=>{
      if(i>=2*SIZE*SIZE) return false;
      const p=center(i), kind=G.CODES[scan.hitKind[i]];
      const d=distanceToBody(p);
      return d>=0.1 && d<=1.5 && ["stone","pedestrian","rider"].includes(kind);
    });
    put("F07",context==="start" && v>0 && v<1.5 && low!==undefined,low===undefined?
      "Chưa thấy vật thấp trong vành đai":`Vật thấp cách thân ${distanceToBody(center(low)).toFixed(2)} m`,{low});

    if(context==="alley"){
      const obstaclePoints=scan.occupied.filter(i=>G.CODES[scan.hitKind[i]]!=="curb").map(center)
        .filter(p=>p[0]>1.8 && p[0]<6 && p[2]<1.65);
      const corridorAt=rate=>[...pathTimes,2.1,2.4].map(t=>futurePose(v,rate,t));
      const blocks=path=>obstaclePoints.find(p=>path.some(s=>hits(toLocal(p,s),[-0.7,2.25,-1.21,1.21])));
      const corridor=corridorAt(yawRate), obstruction=blocks(corridor);
      const alternatives=[-0.12,0.12,-0.24,0.24].find(delta=>!blocks(corridorAt(yawRate+delta)));
      const hint=alternatives===undefined?"chưa có hướng đã kiểm":`thử đổi hướng ${(alternatives*1.8*180/Math.PI).toFixed(0)}°`;
      put("F08",obstruction,obstruction?`Cản tại x=${obstruction[0].toFixed(1)} m · ${hint}`:
        "Chưa có hit trong hành lang",{obstruction,corridor});
    } else {
      put("F08",false,"Mở kịch bản ngõ hẹp",{corridor:[]});
    }

    const swing=sweep.filter(s=>Math.abs(s.yaw)>0.21).map(s=>({...s,nose:toWorld([2.1,0,0.08],s)}));
    const swingPoint=v<0?scan.occupied.map(center).find(p=>p[2]<1.65 && distanceToBody(p)>0.12 &&
      swing.some(s=>hits(toLocal(p,s),[0.85,2.23,-1.08,1.08]))):null;
    put("F09",v<0 && Math.abs(yawRate)>0.21 && swingPoint,swingPoint?"Đầu xe quét qua vật cản":"Chưa có vật cản ở đầu quỹ đạo",{swingPoint,swing});

    const slot=findSlot(scan);
    put("F10",slot,slot?`Ứng viên ${slot.length.toFixed(1)} × ${slot.width.toFixed(1)} m · ${Math.round(slot.coverage*100)}% ô có tia`:
      "Chưa đủ bằng chứng cho ô trống giữa hai cụm",{slot});
    const failing=health.map((h,i)=>h.gated?G.CAMS[i].name:null).filter(Boolean);
    put("F11",failing.length,failing.length?`Tách camera ${failing.join(", ")} khỏi hợp nhất`:"Bốn camera đang tham gia hợp nhất");
    put("F12",ttc!==null && ttc<1.2,ttc!==null?`TTC dự báo ${ttc.toFixed(2)} s`:"Theo dõi sự kiện TTC dưới 1,2 s",{ttc});
    return result;
  }

  function findSlot(scan) {
    const width=2.2, required=5.0;
    for(const sign of [-1,1]) {
      const status=[];
      for(let x=-6.9;x<=6.9;x+=STEP) {
        let seen=0,total=0,blocked=false;
        for(let d=1.28;d<1.28+width-0.05;d+=STEP) for(let z=0;z<Z.length-1;z++){
          const i=indices([x,sign*d,(Z[z]+Z[z+1])/2]);
          const cell=scan.cells[i];
          if(cell===OCCUPIED) blocked=true;
          if(cell===FREE) seen++;
          total++;
        }
        status.push({x,blocked,ratio:seen/total});
      }
      const posts=status.map((s,i)=>s.blocked?i:-1).filter(i=>i>=0);
      for(let i=1;i<posts.length;i++){
        const clear=status.slice(posts[i-1]+1,posts[i]);
        const length=clear.length*STEP;
        if(length<required) continue;
        const coverage=clear.reduce((a,s)=>a+s.ratio,0)/clear.length;
        if(coverage<0.58 || clear.some(s=>s.ratio<0.3)) continue;
        return {x:clear[0].x-STEP/2,y:sign*1.28,otherY:sign*(1.28+width),length,width,coverage};
      }
    }
    return null;
  }

  class Replay {
    constructor(key) { this.reset(key); }
    reset(key) {
      this.key=key;
      this.faults=G.CAMS.map(()=>[]);
      this.frames=new Map();
    }
    fault(index,type,time) {
      this.faults[index]=this.faults[index].filter(e=>e.at<time).concat({type,at:time});
      for(const i of this.frames.keys()) if(i*DT>=time) this.frames.delete(i);
    }
    get(index) {
      index=clamp(Math.round(index),0,MAX_FRAME);
      if(this.frames.has(index)) return this.frames.get(index);
      const first=this.frames.has(index-1)?index:Math.max(0,index-(this.key==="under"?29:3));
      let previous=this.frames.get(first-1);
      for(let i=first;i<=index;i++){
        if(this.frames.has(i)) { previous=this.frames.get(i); continue; }
        const t=i*DT, start=performance.now(), world=sceneAt(this.key,t), health=healthAt(this.faults,t);
        const images=G.capture(world.pose,world.objects,health);
        const begin=performance.now(), scan=G.reconstruct(images,health);
        const reconstructMs=performance.now()-begin;
        const tracks=trackDetections(scan.detections,previous,world.pose);
        const memory=memoryFor(scan,previous,world.pose,t);
        const frame={t,index,scan,images,health,tracks,memory,...world,context:this.key,reconstructMs};
        frame.features=riskFor(frame);
        frame.surfaceError=G.surfaceError(scan.occupied,world.objects,world.pose);
        frame.ms=performance.now()-start;
        this.frames.set(i,frame);
        previous=frame;
      }
      return this.frames.get(index);
    }
    clip(around) {
      const start=clamp(around-Math.round(2.5/DT),0,MAX_FRAME-40);
      const frames=[];
      let riskFrames=0;
      for(let i=start;i<=start+40;i++){
        const f=this.get(i);
        const state=[];
        f.scan.cells.forEach((v,k)=>{
          if(k && state[state.length-2]===v) state[state.length-1]++;
          else state.push(v,1);
        });
        riskFrames+=Number(f.features.F12.active);
        frames.push({t:f.t,state,tracks:f.tracks,ttc:f.features.F12.ttc,pose:f.pose});
      }
      return {format:"omnishield-replay/1",source:"synthetic",scene:this.key,
        grid:{size:SIZE,resolution:STEP,z:Z},intervalSeconds:DT,
        score:Math.max(0,100-Math.round(100*riskFrames/frames.length)),
        scoreDefinition:"100 trừ tỷ lệ phần trăm frame có TTC mô phỏng dưới 1,2 s",
        frames};
    }
  }
  const api={FEATURES,SCENES,DT,DURATION,MAX_FRAME,EGO,Replay,sceneAt,healthAt,riskFor};
  if (typeof module !== "undefined") module.exports=api;
  else root.ReplayCore=api;
})(globalThis);
